from marlin_pytorch import Marlin
import torch
import torch.nn as nn
import torch.nn.functional as F

class MarlinModel(nn.Module):
    def __init__(self, fine_tune, proj_size):
        super().__init__()
        self.fine_tune = fine_tune

        if self.fine_tune:
            self.marlin = Marlin.from_online('marlin_vit_base_ytf').encoder

            self.marlin_projector = nn.Linear(768, proj_size)

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

            video_raw_feat = torch.stack(video_outputs) # (bz, 768)
        else:
            sum_x = video_x.sum(dim=1)
            video_raw_feat = sum_x / batch['seq_length'].unsqueeze(1).float()

        video_feat = self.marlin_projector(video_raw_feat)
        return video_feat
    
class MarlinForClassification(nn.Module):
    def __init__(self, num_classes, fine_tune, proj_size, hidden_size, n_classes):
        super().__init__()
        self.marlin = MarlinModel(num_classes, fine_tune, proj_size)
        self.fc = nn.Sequential(
            nn.Linear(proj_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

    def forward(self, batch):
        video_feat = self.marlin(input)
        logits = self.fc(video_feat)
        return logits