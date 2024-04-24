from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from utils.dataset import *

from torchmetrics import Accuracy, F1Score

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config


class EarlyFusion(LightningModule):

    def __init__(self, 
                n_classes,
                input_size,
                hidden_size,
                lr,
                weight_decay,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.n_classes = n_classes

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes)
        )
        
        self.lr = lr
        self.weight_decay = weight_decay
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.trainer.max_epochs) 
        return [optimizer], [scheduler]


    def forward(self, batch):
        video = batch['video']
        sum_video = video.sum(dim=1)
        video_feat = sum_video / batch['seq_length'].unsqueeze(1).float()

        audio_feat = batch['audio']

        feat = torch.cat([video_feat, audio_feat], dim=1)
            
        output = self.fc(feat)
        return batch['label'], output

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
