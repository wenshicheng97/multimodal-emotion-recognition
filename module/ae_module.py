from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from models.autoencoder import *
from utils.utils import *
from utils.dataset import *

class AutoEncoderModule(LightningModule):
    FEATURES = {
        'gaze_angle': 2,
        'gaze_landmarks': 112,
        'face_landmarks': 136,
        'audio': 128 }
    
    def __init__(self, 
                hidden_sizes, 
                feature, 
                window, 
                stride, 
                normalizer, 
                lr, 
                weight_decay):
        super().__init__()
        self.save_hyperparameters()
        feature_dim = self.FEATURES[feature]
        input_dim = feature_dim * window
        self.autoencoder = AutoEncoder(input_dim, hidden_sizes)
        self.feature = feature
        self.window = window
        self.stride = stride
        self.normalizer = normalizer
        self.lr = lr
        self.weight_decay = weight_decay

    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    
    def forward(self, batch):
        feature_batch = batch[self.feature]
        bz = feature_batch.size(0)

        feature_batch = self.normalizer.minmax_normalize(feature_batch, self.feature)
        feature_transformed, seq_lengths = window_transform(feature_batch, batch['seq_length'], self.window, self.stride)

        max_seq_length = feature_transformed.size(1)
        range_tensor = torch.arange(0, max_seq_length).unsqueeze(0)

        attention_mask = (range_tensor < seq_lengths.unsqueeze(1)).to(self.device)
        attention_mask = attention_mask.int().unsqueeze(-1)

        feature_transformed_reshaped = feature_transformed.view(bz * max_seq_length, -1)
        reconstructed = self.autoencoder(feature_transformed_reshaped).view(bz, max_seq_length, -1)
        reconstructed = reconstructed * attention_mask

        return feature_transformed, reconstructed

    
    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    
    def validation_step(self, batch, batch_idx):
        (y, y_hat) = self(batch)
        loss = F.mse_loss(y_hat, y, reduction='mean')
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 5)
    normalizer = CREMAD_Normalizer(train_loader)
    model = AutoEncoderModule(hidden_sizes=[512], 
                        feature='gaze_angle', 
                        window=5, 
                        stride=2, 
                        normalizer=normalizer, 
                        lr=0.001, 
                        weight_decay=0.0001)
    
    trainer = pl.Trainer(accelerator = 'gpu',
                         max_epochs=10, 
                         devices=4, 
                         strategy='ddp')
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
