import yaml, wandb, os
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning import seed_everything
from utils.dataset import get_dataloader
from module.early_fusion import EarlyFusion
from utils.name import get_search_hparams, get_experiment_name

def train_marlin():
    wandb.init(entity='west-coast', project='emotion-recognition')
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)

    train_loader, val_loader, test_loader = get_dataloader(data='cremad', 
                                                           batch_size=hparams.batch_size,
                                                           fine_tune=hparams.fine_tune)

    model = EarlyFusion(n_classes=hparams.n_classes,
                        input_size=hparams.input_size,
                        hidden_size=hparams.hidden_size,
                        proj_size=hparams.proj_size,
                        lr=hparams.lr,
                        weight_decay=hparams.weight_decay,
                        fine_tune=hparams.fine_tune)

    # wandb name
    name = get_experiment_name(search_hparams, hparams)
    wandb_logger.experiment.name = name

    # checkpoint
    checkpoint_path = f'./{hparams.ckpt}/{name}'

    os.makedirs(checkpoint_path, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        mode='max',
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
    )

    checkpoint_callback_every_n = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='{epoch}-{val_accuracy}',
        every_n_epochs=1,
        save_top_k=-1,
    )

    # training
    trainer = pl.Trainer(accelerator = 'gpu', 
                         max_epochs=hparams.epoch, 
                         logger=wandb_logger, 
                         devices=hparams.devices, 
                         strategy='ddp_find_unused_parameters_true',
                         callbacks=[checkpoint_callback, checkpoint_callback_every_n],
                         num_sanity_val_steps=0,
                         precision=16)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()


if __name__ == '__main__':
    with open('cfgs/earlyfuse.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    global search_hparams
    search_hparams = get_search_hparams(sweep_config)

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train_marlin)