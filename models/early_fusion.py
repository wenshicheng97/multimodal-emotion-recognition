import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hubert import HubertBase
from models.marlin import MarlinModel

class EarlyFusion(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.marlin = MarlinModel(fine_tune=kwargs['fine_tune'], proj_size=kwargs['proj_size'], marlin_model=kwargs['marlin_model'])
        self.hubert = HubertBase(proj_size=kwargs['proj_size'], freeze=(not kwargs['fine_tune']))

        self.classifier = nn.Sequential(
            nn.Linear(kwargs['input_size'], kwargs['hidden_size']),
            nn.ReLU(),
            nn.Linear(kwargs['hidden_size'], kwargs['n_classes'])
        )

    def forward(self, batch):
        video_feat = self.marlin(batch) 

        audio_feat = self.hubert(batch['audio'])
        feat = torch.cat([video_feat, audio_feat], dim=1)
        output = self.classifier(feat)

        return output