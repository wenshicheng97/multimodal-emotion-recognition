from lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from models.transformer import *
from models.autoencoder import *
from utils.utils import *
from utils.dataset import *

from torchmetrics import Accuracy, Precision, Recall, F1Score

class TransformerModule(LightningModule):
    FEATURES = {
        'gaze_angle': 2,
        'gaze_landmarks': 112,
        'face_landmarks': 136,
        'audio': 128 }
    
    def __init__(self, 
                window,
                stride,
                frozen,
                normalizer,
                lr,
                weight_decay,
                **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.window = window
        self.stride = stride

        self.features = ['gaze_landmarks', 'face_landmarks', 'audio']
        self.normalizer = normalizer
        self.lr = lr
        self.weight_decay = weight_decay

        self.encoders = nn.ModuleDict({'gaze_landmarks': torch.load(f"./pretrained_encoder/gaze_landmarks_encoder.pth"),
                                       'face_landmarks': torch.load(f"./pretrained_encoder/face_landmarks_encoder.pth"),
                                       'audio': torch.load(f"./pretrained_encoder/audio_encoder.pth")})
        
        # load from pretrained
        for feature in self.features:
            if frozen:
                freeze(self.encoders[feature])


        self.transformer = MultiModalTransformer(**kwargs)
        
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.lr, 
                                     weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
        #                                                        T_max=self.trainer.max_epochs) 
        return [optimizer]
    
    def forward(self, batch):
        bz = batch['audio'].size(0)
        seq_lengths = batch['seq_length']
        stacked_input = []

        for feature in self.features:
            feature_batch = batch[feature]
            feature_batch = self.normalizer.minmax_normalize(feature_batch, feature)

            # do 1-d CNN
            feature_transformed, new_seq_lengths = window_transform(feature_batch, seq_lengths, self.window, self.stride)
            max_seq_length = feature_transformed.size(1)
            
            # feature_transformed: (batch_size, max_seq_len, feature_size * window)
            feature_transformed_reshaped = feature_transformed.view(bz * max_seq_length, -1)
            encoded = self.encoders[feature](feature_transformed_reshaped).view(bz, max_seq_length, -1)
            
            stacked_input.append(encoded) # (bz, max_seq_len, hidden_size)
            
        #stacked_input: (batch_size, n_modalities, max_seq_len, hidden_size)
        stacked_input = torch.stack(stacked_input, dim=1)
        range_tensor = torch.arange(0, max_seq_length).unsqueeze(0)

        attention_mask = (range_tensor >= new_seq_lengths.unsqueeze(1)).int().to(self.device)
        output = self.transformer(stacked_input, attention_mask) # (bz, n_labels)

        return batch['label'], output


    def on_train_start(self):
        self.train_accuracy = Accuracy(task="multiclass", num_classes=6).to(self.device)
        self.train_f1 = F1Score(task="multiclass", num_classes=6).to(self.device)

    
    def training_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        loss = F.cross_entropy(y_hat, y, reduction='mean')
        self.log('train_loss', loss, on_epoch=True, sync_dist=True)
        self.train_accuracy.update(y_hat, y)
        self.train_f1.update(y_hat, y)
        return loss
    
    
    def on_train_epoch_end(self):
        self.log('train_accuracy', self.train_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.log('train_f1', self.train_f1.compute(), on_epoch=True, sync_dist=True)
        self.train_accuracy.reset()
        self.train_f1.reset()
    

    def on_validation_start(self):
        self.val_accuracy = Accuracy(task="multiclass", num_classes=6).to(self.device)
        self.val_f1 = F1Score(task="multiclass", num_classes=6).to(self.device)


    def validation_step(self, batch, batch_idx):
        y, y_hat = self(batch)
        self.val_accuracy.update(y_hat, y)
        self.val_f1.update(y_hat, y)

    def on_validation_epoch_end(self):
        self.log('val_accuracy', self.val_accuracy.compute(), on_epoch=True, sync_dist=True)
        self.log('val_f1', self.val_f1.compute(), on_epoch=True, sync_dist=True)
        self.val_accuracy.reset()
        self.val_f1.reset()
        
        


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloader('cremad', 5)
    normalizer = CREMAD_Normalizer(train_loader)
    model = TransformerModule(window=10,
                              stride=5,
                              frozen=True,
                              normalizer=normalizer,
                              lr=3e-4,
                              weight_decay=1e-5,
                              d_model=512,
                              n_head=16,
                              d_hid=512,
                              n_layers=24,
                              n_labels=6,
                              dropout=0.2,
                              n_positions=128,
                              n_modalities=3,
                              t_encode=True,
                              lstm_hid=128)
    
    trainer = pl.Trainer(accelerator = 'gpu',
                         max_epochs=20, 
                         devices=1, 
                         strategy='ddp')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
