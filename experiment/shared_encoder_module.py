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

        self.model = SharedEncoder(**kwargs)


    def forward(self, batch):
        return self.model(batch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.trainer.max_epochs) 
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = output.loss
        self.log('train_loss/total_loss', loss, on_epoch=True, sync_dist=True, batch_size=batch['label'].size(0))
        self.log('train_loss/multimodal_loss', output.multimodal_loss, on_epoch=True, sync_dist=True)
        self.log('train_loss/audio_loss', output.audio_loss, on_epoch=True, sync_dist=True)
        self.log('train_loss/video_loss', output.video_loss, on_epoch=True, sync_dist=True)
        self.log('train_loss/text_loss', output.text_loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        # self.audio_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        # self.video_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)
        # self.text_accuracy = Accuracy(task="multiclass", num_classes=self.n_classes).to(self.device)

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        real = output.label
        pred = output.multimodal_logits
        # pred_audio = output.audio_logits
        # pred_video = output.video_logits
        # pred_text = output.text_logits
        self.val_accuracy.update(pred, real)
        # self.audio_accuracy.update(pred_audio, real)
        # self.video_accuracy.update(pred_video, real)
        # self.text_accuracy.update(pred_text, real)


    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        # self.log('audio_accuracy', self.audio_accuracy.compute(), on_epoch=True, sync_dist=True)
        # self.log('video_accuracy', self.video_accuracy.compute(), on_epoch=True, sync_dist=True)
        # self.log('text_accuracy', self.text_accuracy.compute(), on_epoch=True, sync_dist=True)