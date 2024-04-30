import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# define the model
class LSTMModel(nn.Module):
    def __init__(self, **kwargs): # input_size: int, output_size: int, hidden_size: int=128):
        super(LSTMModel, self).__init__()

        self.input_size = kwargs['input_size']
        self.hidden_size = kwargs['hidden_size']

        # the LSTM model
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True, bidirectional=True, num_layers=2)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, kwargs['n_classes'])
        )

        self.feature = kwargs['feature']
    
    def forward(self, batch):
        feature_batch = batch[self.feature]
        seq_length = batch['seq_length']
        
        packed_out = pack_padded_sequence(feature_batch, seq_length.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_out)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.classifier(hidden)
        return out