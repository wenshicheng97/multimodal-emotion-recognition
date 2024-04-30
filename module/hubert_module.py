from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from models.hubert import HubertForClassification
from utils.utils import *
from utils.dataset import *

from torchmetrics import Accuracy, F1Score

class HuBERTModule(LightningModule):
    FEATURES = {
        'gaze_angle': 2,
        'gaze_landmarks': 112,
        'face_landmarks': 136,
        'spectrogram': 128,
        'au17': 17,
        'au18': 18 }
    
    def __init__(self,
                num_labels,
                lr, 
                weight_decay):
        super().__init__()
        self.save_hyperparameters(ignore=['normalizer'])
        self.feature = 'audio'
        self.hubert = HubertForClassification(num_labels=num_labels)
        self.num_labels = num_labels

        self.lr = lr
        self.weight_decay = weight_decay

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    
    def forward(self, batch):
        feature_batch = batch[self.feature]
        label = batch['label']
        
        output = self.hubert(feature_batch)

        return label, output

    
    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_labels).to(self.device)
        # self.val_f1 = F1Score(task="multiclass", num_classes=self.num_labels).to(self.device)


    def validation_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        self.val_accuracy.update(y_hat, y)
        # self.val_f1.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        # self.log('val_f1', self.val_f1.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()
        # self.val_f1.reset()


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 5)
    model = HuBERTModule(num_labels=6,
                         lr=0.001, 
                         weight_decay=0.0001)
    
    trainer = pl.Trainer(accelerator = 'gpu',
                         max_epochs=10, 
                         devices=4, 
                         strategy='ddp')
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
