import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hubert import HubertBase
from models.marlin import MarlinModel
from models.bert import BertBase


class BimodalGatedMultimodalUnit(nn.Module):
    def __init__(self, dim_h1, dim_h2, dim_out):
        super().__init__()
        self.fc_h1 = nn.Linear(dim_h1, dim_out)
        self.fc_h2 = nn.Linear(dim_h2, dim_out)
        self.fc_z1 = nn.Linear(dim_h1, dim_out)
        self.fc_z2 = nn.Linear(dim_h2, dim_out)

    def forward(self, h1, h2):
        h1_transformed = self.fc_h1(h1)
        h2_transformed = self.fc_h2(h2)

        z1 = torch.sigmoid(self.fc_z1(h1))
        z2 = torch.sigmoid(self.fc_z2(h2))

        z = z1 + z2
        z = z / torch.sum(z, dim=1, keepdim=True)

        h = z1 * h1_transformed + z2 * h2_transformed
        return h
    

class TrimodalGatedMultimodalUnit(nn.Module):
    def __init__(self, dim_h1, dim_h2, dim_h3, dim_out):
        super().__init__()
        self.fc_h1 = nn.Linear(dim_h1, dim_out)
        self.fc_h2 = nn.Linear(dim_h2, dim_out)
        self.fc_h3 = nn.Linear(dim_h3, dim_out)

        self.fc_z1 = nn.Linear(dim_h1, dim_out)
        self.fc_z2 = nn.Linear(dim_h2, dim_out)
        self.fc_z3 = nn.Linear(dim_h3, dim_out)

    def forward(self, h1, h2, h3):
        h1_transformed = self.fc_h1(h1)
        h2_transformed = self.fc_h2(h2)
        h3_transformed = self.fc_h3(h3)

        z1 = torch.sigmoid(self.fc_z1(h1))
        z2 = torch.sigmoid(self.fc_z2(h2))
        z3 = torch.sigmoid(self.fc_z3(h3))

        z = z1 + z2 + z3
        z = z / torch.sum(z, dim=1, keepdim=True)

        h = z1 * h1_transformed + z2 * h2_transformed + z3 * h3_transformed
        return h
    
    
class GeneralizedGatedMultimodalUnit(nn.Module):
    def __init__(self, dims, dim_out):
        super().__init__()
        self.modalities = len(dims)
        self.fc_transform = nn.ModuleList([nn.Linear(dim, dim_out) for dim in dims])
        # self.fc_gates = nn.ModuleList([nn.Linear(dim, dim_out) for dim in dims])
        self.fc_gates = nn.ModuleList([nn.Linear(dim * self.modalities, dim_out) for dim in dims])
        self.epsilon = 1e-8
        self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, *inputs):
        assert len(inputs) == self.modalities, "Number of inputs must match number of modalities"
        
        # Concatenate gate
        all_input = torch.cat(inputs, dim=1)
        transformed = [self.fc_transform[i](inputs[i]) for i in range(self.modalities)]
        activated = [torch.tanh(transformed[i]) for i in range(self.modalities)]
        gates = [torch.sigmoid(self.fc_gates[i](all_input)) for i in range(self.modalities)]
        total_gates = torch.stack(gates, dim=0).sum(dim=0)
        normalized_gates = [gate / total_gates for gate in gates]
        fused_output = torch.stack([normalized_gates[i] * activated[i] for i in range(self.modalities)], dim=0).sum(dim=0)
        skip_connection = torch.stack(transformed, dim=0).mean(dim=0)
        combined_output = fused_output + skip_connection
        final_output = self.layer_norm(combined_output)

        # Single gate
        # transformed = [self.fc_transform[i](inputs[i]) for i in range(self.modalities)]
        # activated = [torch.tanh(transformed[i]) for i in range(self.modalities)]
        # gates = [torch.sigmoid(self.fc_gates[i](inputs[i])) for i in range(self.modalities)]
        # total_gates = torch.stack(gates, dim=0).sum(dim=0)
        # normalized_gates = [gate / total_gates for gate in gates]
        # fused_output = torch.stack([normalized_gates[i] * activated[i] for i in range(self.modalities)], dim=0).sum(dim=0)
        # skip_connection = torch.stack(transformed, dim=0).mean(dim=0)
        # combined_output = fused_output + skip_connection
        # final_output = self.layer_norm(combined_output)
        
        return final_output
    

class GatedFusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dims = []
        self.marlin = MarlinModel(fine_tune=kwargs['fine_tune'], proj_size=kwargs['proj_size'], marlin_model=kwargs['marlin_model'])
        dims.append(kwargs['proj_size'])
        self.hubert = HubertBase(proj_size=kwargs['proj_size'], freeze=(not kwargs['fine_tune']))
        dims.append(kwargs['proj_size'])
        if kwargs['data'] == 'mosei':
            self.bert = BertBase(proj_size=kwargs['proj_size'], freeze=(not kwargs['fine_tune']))
            print(kwargs['data'])
            dims.append(kwargs['proj_size'])

        self.fusion_model = GeneralizedGatedMultimodalUnit(dims=dims, dim_out=kwargs['dim_out'])
        self.classifier = nn.Sequential(
            nn.Linear(kwargs['dim_out'], kwargs['hidden_size']),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_size'], kwargs['n_classes'])
        )

    def forward(self, batch):
        video_feat = self.marlin(batch) # (bz, 256)

        audio_feat = self.hubert(batch['audio']) # (bz, 256)

        input = [video_feat, audio_feat]
        if hasattr(self, 'bert'):
            text_feat = self.bert(batch['text'])
            input.append(text_feat)
        
        fused_output = self.fusion_model(*input)

        return self.classifier(fused_output)
