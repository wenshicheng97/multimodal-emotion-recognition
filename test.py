import torch, os
import torch.nn as nn

from utils.dataset import get_dataloader
from module.marlin_module import MarlinModule
from module.hubert_module import HuBERTModule

from torchmetrics import Accuracy

marlin_path = '/scratch1/tangyimi/emotion/ckpt/marlin_small/best.ckpt'
hubert_path = '/scratch1/tangyimi/emotion/ckpt/hubert/lr5e-05/best.ckpt'

def late_fusion():
    marlin_model = MarlinModule.load_from_checkpoint(marlin_path)
    hubert_model = HuBERTModule.load_from_checkpoint(hubert_path)

    marlin_model.eval()
    hubert_model.eval()

    _, val_loader, _ = get_dataloader('cremad',
                                      batch_size=2,
                                      fine_tune=True)
    device = marlin_model.device

    accuracy_metric = Accuracy(task="multiclass", num_classes=6).to(device)

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            y, marlin_output = marlin_model(batch)
            _, hubert_output = hubert_model(batch)

            marlin_output_prob, hubert_output_prob = torch.softmax(marlin_output, dim=1), torch.softmax(hubert_output, dim=1)
            output = marlin_output_prob + hubert_output_prob

            accuracy_metric.update(output, y)

    print(f'Accuracy: {accuracy_metric.compute()}')

if __name__ == '__main__':
    late_fusion()
