import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len :int = 4096):
        super().__init__()
        self.d_model = d_model
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape (batch_size, n_modalities, seq_len, embedding_dim)
        Returns:
            Positional encoded x
        """
        batch_size, n_modalities, seq_len, _ = x.size()
        return x + self.pe[:seq_len].squeeze().repeat(batch_size, n_modalities, 1, 1)
    

class MultiModalTransformer(nn.Module):
    def __init__(self, d_model: int, 
                 n_head: int, 
                 d_hid: int, 
                 n_layers: int,
                 n_labels: int,
                 dropout: float = 0.5,
                 n_positions: int = 512,
                 n_modalities: int = 3,
                 t_encode: bool = False,
                 lstm_hid: int = 128,
                 use_cls: bool = False,
                 cls_token_init: torch.Tensor = None):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.t_encode = t_encode
        self.use_cls = use_cls
        if not t_encode:
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            self.t_encoder = nn.Embedding(n_positions + 1, d_model, padding_idx=0)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model, 
                                                                              n_head, 
                                                                              d_hid, 
                                                                              dropout, 
                                                                              batch_first=True),
                                                      n_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) if cls_token_init is None else nn.Parameter(cls_token_init)
        if not use_cls:
            self.lstm = nn.LSTM(input_size=d_hid * n_modalities, hidden_size=lstm_hid, batch_first=True, bidirectional=True, num_layers=1)
            self.fc1 = nn.Linear(lstm_hid * 2, lstm_hid)
        else:
            self.fc1 = nn.Linear(d_hid, lstm_hid)
        self.fc2 = nn.Linear(lstm_hid, n_labels)
        
    def forward(self, batch, attn_mask):
        """
        Arguments:
            batch: Tensor, shape (batch_size, n_modalities, seq_len, embedding_dim)
            attn_mask: Tensor, shape (batch_size, seq_len),
                attention mask for trasformer layer, 0 for unmasked, 1 for masked
        Returns:
            out: Tensor, shape (batch_size, n_labels) logits of last linear layer
        """
        device = batch.device
        batch_size, n_modalities, seq_len, dim = batch.size()
        lengths = (~attn_mask.bool()).sum(dim=1)
        attn_mask = attn_mask.bool().repeat_interleave(self.n_modalities, dim=1)
        
        if not self.t_encode:
            out = self._combine_modality(self.pos_encoder(batch))
        else:
            out = torch.arange(1, seq_len + 1).to(device)
            out = out.repeat_interleave(self.n_modalities).unsqueeze(0).repeat(batch_size, 1)
            out = self.t_encoder(out) + self._combine_modality(batch)
        
        if self.use_cls:
            cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
            out = torch.cat([cls_tokens, out], dim=1)
            attn_mask = torch.cat([torch.zeros(batch_size, 1).to(device).bool(), attn_mask], dim=1)
        
        out = self.transformer_encoder(out, src_key_padding_mask=attn_mask)

        if self.use_cls:
            cls_token_representation = out[:, 0, :]
            out = self.fc1(cls_token_representation)
        else:
            out = out.reshape(batch_size, seq_len, n_modalities, -1).reshape(batch_size, seq_len, -1)
            packed_out = pack_padded_sequence(out, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed_out)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            out = self.fc1(hidden)
        out = F.relu(out)
        out = self.fc2(out)
        return out
    
    @staticmethod
    def _combine_modality(x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(1) * x.size(2), -1).squeeze(1)
        