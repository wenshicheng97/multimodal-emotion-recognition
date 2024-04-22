from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from models.lstm import *
from utils.utils import *
from utils.dataset import *

from torchmetrics import Accuracy, F1Score

from marlin_pytorch import Marlin
from marlin_pytorch.config import resolve_config


class MarlinModule(LightningModule):
    EMBED_DIM = {'marlin_vit_small_ytf': 384, 
                'marlin_vit_base_ytf': 768, 
                'marlin_vit_large_ytf': 1024}

    def __init__(self, 
                model,
                n_classes,
                hidden_size,
                fine_tune,
                lr,
                weight_decay,
                ):
        super().__init__()
        self.save_hyperparameters()

        self.fine_tune = fine_tune
        if fine_tune: 
            self.model = Marlin.from_online(model).encoder
        self.feature = 'video'

        self.n_classes = n_classes

        self.fc = nn.Sequential(
            nn.Linear(self.EMBED_DIM[model], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_classes))
        self.lr = lr
        self.weight_decay = weight_decay
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.trainer.max_epochs) 
        return [optimizer], [scheduler]

    # @classmethod
    # def from_module(cls, model, learning_rate: float = 1e-4, distributed=False):
    #     return cls(model, learning_rate, distributed)

    def forward(self, batch):
        x = batch[self.feature]

        if self.fine_tune:
            feat = self.model.extract_features(x, True)
        else:
            sum_x = x.sum(dim=1)
            feat = sum_x / batch['seq_length'].unsqueeze(1).float()
        output = self.fc(feat)
        return batch['label'], output

    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.n_classes).to(self.device)


    def validation_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        self.val_accuracy.update(y_hat, y)
        self.val_f1.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()
        self.val_f1.reset()
