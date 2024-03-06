from module.ae_module import *
import lightning.pytorch as pl
import wandb
from lightning.pytorch.loggers import WandbLogger
import yaml
import argparse
from lightning import seed_everything
from utils.dataset import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature', type=str, required=True)
    return parser.parse_args()


def train_autoencoder():
    args = get_args()
    wandb.init()
    wandb_logger = WandbLogger(entity='west-coast', project='emotion-recognition')
    hparams = wandb.config

    seed_everything(hparams.seed)
    
    # load data
    train_loader, val_loader, test_loader = get_dataloader('cremad', hparams.batch_size)

    # normalize
    normalizer = CREMAD_Normalizer(train_loader)

    # load model
    model = AutoEncoderModule(hidden_sizes=hparams.hidden_sizes, 
                        feature=args.feature, 
                        window=hparams.window, 
                        stride=hparams.stride, 
                        normalizer=normalizer, 
                        lr=hparams.lr, 
                        weight_decay=hparams.weight_decay)

    # wandb name
    wandb_logger.experiment.name = f'{args.feature}_lr{hparams.lr}_hidden{hparams.hidden_sizes}'

    trainer = pl.Trainer(accelerator = 'gpu', 
                         max_epochs=hparams.epoch, 
                         logger=wandb_logger, 
                         devices=hparams.devices, 
                         strategy='ddp')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()

    Path(f'./sample/{args.feature}').mkdir(exist_ok=True, parents=True)
    torch.save(model.autoencoder.encoder, f'./pretrained_encoders/{args.feature}/lr{hparams.lr}_hidden{hparams.hidden_sizes}_encoder.pth')

if __name__ == '__main__':
    with open('cfgs/autoencoder.yaml', 'r') as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, entity='west-coast',project="emotion-recognition")

    wandb.agent(sweep_id, function=train_autoencoder)
