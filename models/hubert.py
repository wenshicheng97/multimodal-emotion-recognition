import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

class HubertBase(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.hubert = HubertForSequenceClassification.from_pretrained("facebook/hubert-base-ls960", num_labels=num_labels)

    def forward(self, input_values):
        outputs = self.hubert(input_values=input_values)
        return outputs.logits
