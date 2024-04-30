import torch
import torch.nn as nn
from transformers import HubertModel

class HubertBase(nn.Module):
    def __init__(self, proj_size=None, freeze=False):
        super().__init__()
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        # self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.hidden_size = self.hubert.config.hidden_size
        self.proj_size = self.hubert.config.classifier_proj_size if proj_size is None else proj_size
        self.projector = nn.Linear(self.hidden_size, self.proj_size)
        self._init_weights(self.projector)
        self.freeze = freeze

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_values):
        if self.freeze:
            with torch.no_grad():
                outputs = self.hubert(input_values)
        else:
            outputs = self.hubert(input_values)
        hidden_states = outputs[0]
        hidden_states = self.projector(hidden_states)
        return hidden_states.mean(dim=1)
    
    
class HubertForClassification(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hubert = HubertBase(kwargs['proj_size'], not kwargs['fine_tune'])
        self.classifier = nn.Linear(self.hubert.proj_size, kwargs['n_classes'])
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch):
        audio = batch['audio']
        hidden_states = self.hubert(audio)
        output = self.classifier(hidden_states)
        return output
