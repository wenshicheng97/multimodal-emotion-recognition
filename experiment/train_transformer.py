from module.tf_module import *
import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
from lightning import seed_everything
from utils.dataset import *


def train_transformer():
    wandb.init()
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)
    
    # load data
    train_loader, val_loader, test_loader = get_dataloader('cremad', hparams.batch_size)

    # normalize
    normalizer = CREMAD_Normalizer(train_loader)

    # load model
    model = TransformerModule(window=hparams.window,
                              stride=hparams.stride,
                              frozen=hparams.frozen,
                              normalizer=normalizer,
                              lr=hparams.lr,
                              weight_decay=hparams.weight_decay,
                              d_model=hparams.d_model,
                              n_head=hparams.n_head,
                              d_hid=hparams.d_hid,
                              n_layers=hparams.n_layers,
                              n_labels=hparams.n_labels,
                              dropout=hparams.dropout,
                              n_ctx=hparams.n_ctx,
                              n_modalities=hparams.n_modalities,
                              t_encode=hparams.t_encode,
                              lstm_hid=hparams.lstm_hid)

    # wandb name
    wandb_logger.experiment.name = f'transformer_lr{hparams.lr}_frozen{hparams.frozen}_t_encode{hparams.t_encode}'

    # checkpoint
    checkpoint_path = Path(f'./checkpoint/lr{hparams.lr}_frozen{hparams.frozen}_t_encode{hparams.t_encode}').mkdir(exist_ok=True, parents=True) 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
    )

    # training
    trainer = pl.Trainer(accelerator = 'gpu', 
                         max_epochs=hparams.epoch, 
                         logger=wandb_logger, 
                         devices=hparams.devices, 
                         strategy='ddp',
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()


    
if __name__ == '__main__':
    with open('cfgs/transformer.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train_transformer)
