import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# define the model
class LSTMModel(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_size: int=128):
    super(LSTMModel, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size

    # the LSTM model
    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True, num_layers=2)
    self.fc1 = nn.Linear(hidden_size*2, hidden_size)
    self.fc2 = nn.Linear(hidden_size, output_size)


  def forward(self, batch, seq_length):
    packed_out = pack_padded_sequence(batch, seq_length.cpu(), batch_first=True, enforce_sorted=False)
    _, (hidden, _) = self.lstm(packed_out)
    hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
    out = self.fc1(hidden)
    out = F.relu(out)
    out = self.fc2(out)
    return out