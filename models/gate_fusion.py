import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hubert import HubertBase
from models.marlin import MarlinModel


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
        self.fc_gates = nn.ModuleList([nn.Linear(dim, dim_out) for dim in dims])

    def forward(self, *inputs):
        assert len(inputs) == self.modalities, "Number of inputs must match number of modalities"
        
        transformed = [self.fc_transform[i](inputs[i]) for i in range(self.modalities)]
        gates = [torch.sigmoid(self.fc_gates[i](inputs[i])) for i in range(self.modalities)]

        total_gates = torch.stack(gates, dim=0).sum(dim=0)
        normalized_gates = [gate / total_gates for gate in gates]

        fused_output = torch.stack([normalized_gates[i] * transformed[i] for i in range(self.modalities)], dim=0).sum(dim=0)
        
        return fused_output
    

class GatedFusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.marlin = MarlinModel(fine_tune=kwargs['fine_tune'], proj_size=kwargs['proj_size'])
        self.hubert = HubertBase(proj_size=kwargs['proj_size'], freeze=(not kwargs['fine_tune']))

        self.fusion_model = GeneralizedGatedMultimodalUnit(dims=[kwargs['proj_size'], kwargs['proj_size']], 
                                                     dim_out=kwargs['dim_out'])
        self.classifier = nn.Sequential(
            nn.Linear(kwargs['dim_out'], kwargs['hidden_size']),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_size'], kwargs['n_classes'])
        )

    def forward(self, batch):
        # video
        video_feat = self.marlin(batch) # (bz, 256)

        audio_feat = self.hubert(batch['audio']) # (bz, 256)

        fused_output = self.fusion_model(video_feat, audio_feat)
        return self.classifier(fused_output)
