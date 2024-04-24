from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from models.hubert import HubertBase
from marlin_pytorch import Marlin
from torchmetrics import Accuracy, F1Score


class EarlyFusion(LightningModule):

    def __init__(self, 
                n_classes,
                input_size,
                hidden_size,
                proj_size,
                lr,
                weight_decay,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes

        self.hubert = HubertBase(num_labels=n_classes, proj_size=proj_size)
        self.marlin = Marlin.from_online('marlin_vit_base_ytf').encoder

        self.marlin_projector = nn.Linear(768, proj_size)

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )

        self.lr = lr
        self.weight_decay = weight_decay


    def forward(self, batch):
        # video
        video_x = batch['video']
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

        video_feat = self.marlin_projector(video_raw_feat) # (bz, 256)

        # audio
        audio_x = batch['audio']
        
        audio_feat = self.hubert(audio_x) # (bz, 256)

        feat = torch.cat([video_feat, audio_feat], dim=0) 
        output = self.fc(feat)
        return batch['label'], output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.trainer.max_epochs) 
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)

    def validation_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        self.val_accuracy.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()
