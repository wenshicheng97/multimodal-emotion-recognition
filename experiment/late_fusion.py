import torch
from tqdm import tqdm

from utils.dataset import get_dataloader
from experiment.lightning_module import ExperimentModule
from lightning import seed_everything
from torchmetrics import Accuracy
from sklearn.metrics import f1_score


def late_fusion(model_paths, data, fine_tune):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    all_models = []

    for model_path in model_paths:
        current_model = ExperimentModule.load_from_checkpoint(model_path)
        current_model = current_model.to(device)
        current_model.eval()
        all_models.append(current_model)

    seed_everything(0)

    _, val_loader, _ = get_dataloader(data=data,
                                      batch_size=32,
                                      fine_tune=fine_tune,
                                      lstm=False)

    accuracy_metric = Accuracy(task="multiclass", num_classes=6).to(device)

    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}

            y = batch['label']

            prob_logits = []

            for model in all_models:
                _, y_hat = model(batch)
                prob_logits.append(torch.softmax(y_hat, dim=1))
            
            prob_logits = torch.stack(prob_logits, dim=0)
            output = torch.sum(prob_logits, dim=0)

            accuracy_metric.update(output, y)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy())

    f1 = f1_score(y_true, y_pred, average='micro')

    print(f'Late fusion Accuracy: {accuracy_metric.compute()} F1: {f1}')

if __name__ == '__main__':
    # mosei ft
    # model_paths = ['/scratch1/tangyimi/emotion/ckpt/mosei/hubert/best.ckpt',
    #                '/scratch1/tangyimi/emotion/ckpt/mosei/bert/best.ckpt',
    #                '/scratch1/tangyimi/emotion/ckpt/mosei/marlin/ft/best.ckpt']
    # cremad ft
    # model_paths = ['/scratch1/tangyimi/emotion/ckpt/cremad/marlin/ft/best.ckpt',
    #                '/scratch1/tangyimi/emotion/ckpt/cremad/hubert/HubertForClassification/best.ckpt']

    # cremad lp
    model_paths = ['/scratch1/tangyimi/emotion/ckpt/cremad/marlin/lp/best.ckpt',
                   '/scratch1/tangyimi/emotion/ckpt/cremad/hubert/lp/best.ckpt']

    late_fusion(model_paths, 'cremad', fine_tune=False)
