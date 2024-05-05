import torch, argparse
from sklearn.metrics import f1_score
from experiment.lightning_module import ExperimentModule
from utils.dataset import get_dataloader
from lightning import seed_everything
from tqdm import tqdm

best ='/home/tangyimi/emotion/ckpt/mosei/sharedencoder/SharedEncoder_lr=0.001/best.ckpt'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=str)
    parser.add_argument('--fine_tune', action='store_true')

    return parser.parse_args()


def evalute_f1(model_path, seed, data, fine_tune, average='macro'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ExperimentModule.load_from_checkpoint(model_path)
    model = model.to(device)
    model.eval()

    seed_everything(seed)

    _, val_loader, _ = get_dataloader(data=data, 
                                      batch_size=32, 
                                      fine_tune=fine_tune, 
                                      lstm=False)

    with torch.no_grad():
        y_true = []
        y_pred = []
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            y, y_hat = model(batch)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_hat.argmax(dim=1).cpu().numpy())

    f1 = f1_score(y_true, y_pred, average=average)

    print(f'Model {type(model.model).__name__} F1 score on {data}: {f1}')


if __name__ == '__main__':
    args = get_args()
    evalute_f1(args.model_path, args.seed, args.data, args.fine_tune)
    