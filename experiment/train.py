import yaml, wandb, os, argparse
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import seed_everything
from utils.dataset import get_dataloader

from utils.name import get_search_hparams, get_experiment_name

from module.lightning_module import ExperimentModule

os.environ['WANDB_SILENT'] = 'true'

def train():
    wandb.init(entity='west-coast', project='emotion-recognition')
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)

    train_loader, val_loader, test_loader = get_dataloader(data='cremad', 
                                                           batch_size=hparams.batch_size,
                                                           fine_tune=hparams.fine_tune)

    model = ExperimentModule(**hparams)

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
    trainer = pl.Trainer(max_epochs=hparams.epoch, 
                         logger=wandb_logger, 
                         strategy=hparams.strategy,
                         callbacks=[checkpoint_callback, checkpoint_callback_every_n],
                         num_sanity_val_steps=0,
                         precision=hparams.precision)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.validate(model, dataloaders=val_loader)

    wandb.finish()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', type=str)
    parser.add_argument('--sweep_name', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    with open(f'cfgs/{args.experiment}.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    global search_hparams
    search_hparams = get_search_hparams(sweep_config)
    sweep_config['name'] += args.sweep_name

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train)