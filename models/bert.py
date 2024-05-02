import torch
import torch.nn as nn
from transformers import BertModel

class BertBase(nn.Module):
    def __init__(self, freeze=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.hidden_size = self.bert.config.hidden_size
        self.hidden_dropout_prob = self.bert.config.hidden_dropout_prob
        self._init_weights()
        self.freeze = freeze

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_values):
        if self.freeze:
            with torch.no_grad():
                outputs = self.bert(input_values)
        else:
            outputs = self.bert(input_values)
        hidden_states = outputs[0]
        return hidden_states.mean(dim=1)
    
    
class BertForClassification(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bert = BertBase(not kwargs['fine_tune'])
        self.dropout = nn.Dropout(self.bert.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.hidden_size, kwargs['n_classes'])
        self._init_weights(self.classifier)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch):
        text = batch['text']
        hidden_states = self.bert(text)
        output = self.classifier(hidden_states)
        return output
