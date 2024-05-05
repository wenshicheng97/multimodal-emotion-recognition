import torch
import torch.nn as nn
from transformers import BertModel

class BertBase(nn.Module):
    def __init__(self, proj_size=None, freeze=False):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.hidden_size = self.bert.config.hidden_size
        self.hidden_dropout_prob = self.bert.config.hidden_dropout_prob
        # self._init_weights()
        self.freeze = freeze

        if proj_size:
            self.projector = nn.Linear(self.hidden_size, proj_size)
            self._init_weights(self.projector)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_values):
        attention_mask = (input_values != 0).long()
        if self.freeze:
            with torch.no_grad():
                outputs = self.bert(input_values, attention_mask=attention_mask)
        else:
            outputs = self.bert(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state[:,0,:]

        if hasattr(self, 'projector'):
            hidden_states = self.projector(hidden_states)

        return hidden_states
    
    
class BertForClassification(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.bert = BertBase(proj_size=kwargs['proj_size'], freeze=not kwargs['fine_tune'])
        ln = nn.Linear(self.bert.hidden_size, kwargs['n_classes'])
        self._init_weights(ln)
        self.classifier = nn.Sequential(
            nn.Dropout(self.bert.hidden_dropout_prob),
            ln
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch):
        text = batch['text']
        hidden_states = self.bert(text)
        output = self.classifier(hidden_states)
        return output
