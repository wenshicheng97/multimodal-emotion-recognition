import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
from lightning import seed_everything
from utils.dataset import *
import argparse
from module.hubert_module import *
import os


def train_hubert():
    wandb.init()
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)

    train_loader, val_loader, test_loader = get_dataloader('cremad', hparams.batch_size)

    model = HuBERTModule(num_labels=hparams.num_labels,
                         lr=hparams.lr,
                         weight_decay=hparams.weight_decay)

    # wandb name

    checkpoint_path = f'./ckpt/hubert/lr{hparams.lr}'
    os.makedirs(checkpoint_path, exist_ok=True)

    # checkpoint
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
                         strategy='ddp_find_unused_parameters_true',
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=0,
                         precision=hparams.precision)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()


if __name__ == '__main__':
    with open('cfgs/hubert.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train_hubert)