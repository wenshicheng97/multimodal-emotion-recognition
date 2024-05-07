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
from models.shared_encoder import SharedEncoder

class ExperimentModule(LightningModule):

    def __init__(self, 
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs['lr']
        self.weight_decay = kwargs['weight_decay']
        self.n_classes = kwargs['n_classes']
        self.use_text = (kwargs['data'] == 'mosei')

        self.model = SharedEncoder(**kwargs)

        self.automatic_optimization = False


    def forward(self, batch, mode='multimodal'):
        return self.model(batch, mode)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # Train for each mode sequentially with their respective losses
        modes = ['audio', 'video', 'text'] if self.use_text else ['audio', 'video']
        losses = {}

        for mode in modes:
            output = self.forward(batch, mode)
            loss = output.loss
            self.log(f'train_loss/{mode}_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
            losses[mode] = loss
            self.manual_backward(loss)
            self.optimizers().step()
            self.optimizers().zero_grad()

        with self.model.train_mode(mode='multimodal'):
            multimodal_output = self.forward(batch, 'multimodal')
            multimodal_loss = multimodal_output.loss
            self.log('train_loss/multimodal_loss', multimodal_loss, on_step=False, on_epoch=True, sync_dist=True)
            losses['multimodal'] = multimodal_loss
            self.manual_backward(multimodal_loss)
            self.optimizers().step()
            self.optimizers().zero_grad()

        total_loss = sum(losses.values())
        self.log('train_loss/total_loss', total_loss, on_step=False, on_epoch=True, sync_dist=True)
        return total_loss
    
    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch, 'multimodal')
        real = output.label
        pred = output.logits
        self.val_accuracy.update(pred, real)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()