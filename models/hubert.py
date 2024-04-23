import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample
from transformers import HubertModel, HubertConfig, HubertForSequenceClassification, Wav2Vec2FeatureExtractor

class HubertBase(nn.Module):
    def __init__(self, proj_size=None):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hidden_size = self.hubert.config.hidden_size
        self.proj_size = self.hubert.config.classifier_proj_size if proj_size is None else proj_size
        self.projector = nn.Linear(self.hidden_size, self.proj_size)
        self._init_weights(self.projector)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_values):
        outputs = self.hubert(input_values)
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        return hidden_states.mean(dim=1)
    
    
class HubertForClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.hubert = HubertBase()
        self.classifier = nn.Linear(self.hubert.proj_size, num_labels)
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_values):
        hidden_states = self.hubert(input_values)
        output = self.classifier(hidden_states)
        return output
