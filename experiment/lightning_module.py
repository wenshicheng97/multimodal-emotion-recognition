from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import Accuracy

from models.gate_fusion import GatedFusion
from models.marlin import MarlinForClassification
from models.hubert import HubertForClassification
from models.early_fusion import EarlyFusion
from models.lstm import LSTMModel
from models.bert import BertForClassification

class ExperimentModule(LightningModule):

    def __init__(self, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']
        self.n_classes = kwargs['n_classes']

        self.model = globals()[kwargs['model']](**kwargs)


    def forward(self, batch):
        return batch['label'], self.model(batch)
    
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
        self.log('train_loss', loss, on_epoch=True, sync_dist=True, batch_size=batch['label'].size(0))
        return loss

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)

    def validation_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        self.val_accuracy.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()