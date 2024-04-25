from marlin_pytorch import Marlin
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarlinModel(nn.Module):
    EMDED_DIM = {'marlin_vit_small_ytf': 384,
                'marlin_vit_base_ytf': 768, 
                'marlin_vit_large_ytf': 1024}

    def __init__(self, fine_tune, proj_size, marlin_model):
        super().__init__()
        self.fine_tune = fine_tune
        self.marlin_model = marlin_model
        self.proj_size = eval(proj_size)

        if self.fine_tune:
            self.marlin = Marlin.from_online(marlin_model).encoder

        if self.proj_size:
            self.marlin_projector = nn.Linear(self.EMDED_DIM[marlin_model], proj_size)

    def forward(self, batch):
        video_x = batch['video']
        if self.fine_tune:
            feat = self.marlin.extract_features(video_x, True)
            start = 0
            video_outputs = []
            
            for num_seg in batch['num_seg']:
                end = start + num_seg
                current_feat = feat[start:end]
                video_feat = torch.mean(current_feat, dim=0)
                video_outputs.append(video_feat)
                start = end

            video_feat = torch.stack(video_outputs) # (bz, 768)
        else:
            sum_x = video_x.sum(dim=1)
            video_feat = sum_x / batch['seq_length'].unsqueeze(1).float()

        if self.proj_size:
            video_feat = self.marlin_projector(video_feat)
        return video_feat
    
class MarlinForClassification(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.marlin = MarlinModel(kwargs['fine_tune'], 
                                kwargs['proj_size'], 
                                kwargs['marlin_model'])
        self.fc = nn.Sequential(
            nn.Linear(self.marlin.EMDED_DIM[self.marlin.marlin_model], kwargs['hidden_size']),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_size'], kwargs['n_classes'])
        )

    def forward(self, batch):
        video_feat = self.marlin(batch)
        logits = self.fc(video_feat)
        return logits