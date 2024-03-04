import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len :int =4096):
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
