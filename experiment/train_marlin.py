import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml
from lightning import seed_everything
from utils.dataset import *
import argparse
from module.marlin_module import *

def train_marlin():
    wandb.init(entity='west-coast', project='emotion-recognition')
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)

    train_loader, val_loader, test_loader = get_dataloader(data='cremad', 
                                                           batch_size=hparams.batch_size,
                                                           fine_tune=hparams.fine_tune,)

    model = MarlinModule(model=hparams.model,
                         n_classes=hparams.n_classes,
                         hidden_size=hparams.hidden_size,
                         fine_tune=hparams.fine_tune,
                         lr=hparams.lr,
                         weight_decay=hparams.weight_decay)

    # wandb name
    wandb_logger.experiment.name = f'marlin_lr{hparams.lr}'

    # checkpoint
    checkpoint_path = f'./{hparams.ckpt_dir}/lr{hparams.lr}'

    Path(checkpoint_path).mkdir(exist_ok=True, parents=True) 
    
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
    with open('cfgs/marlin.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train_marlin)