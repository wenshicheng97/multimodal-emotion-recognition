import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from model.layers import PositionalEncoding


class MultiModalTransformer(nn.Module):
    def __init__(self, d_model: int, 
                 n_head: int, 
                 d_hid: int, 
                 n_layers: int, 
                 dropout: float = 0.5,
                 max_seq_len: int = 512,
                 n_modalities: int = 3,
                 t_encode = False):
        super().__init__()
        self.d_model = d_model
        self.n_modalities = n_modalities
        self.t_encode = t_encode
        if not t_encode:
            self.pos_encoder = PositionalEncoding(d_model)
        else:
            self.t_encoder = nn.Embedding(max_seq_len + 1, d_model, padding_idx=0)
        self.transformer_encoder = TransformerEncoder(TransformerEncoderLayer(d_model, 
                                                                              n_head, 
                                                                              d_hid, 
                                                                              dropout, 
                                                                              batch_first=True),
                                                      n_layers)
        
    def forward(self, batch, attn_mask):
        """
        Arguments:
            batch: Tensor, shape (batch_size, n_modalities, seq_len, embedding_dim)
            attn_mask: Tensor, shape (batch_size, seq_len),
                attention mask for trasformer layer, 0 for unmasked, 1 for masked
        Returns:
            out: Tensor, shape (batch_size, N_modalities * seq_len, hidden_dim)
                hidden states of last attention layer
        """
        device = batch.device
        batch_size, n_modalities, seq_len, _ = batch.size()
        attn_mask = attn_mask.bool().repeat_interleave(self.n_modalities, dim=1)
        if not self.t_encode:
            out = self._combine_modality(self.pos_encoder(batch))
        else:
            out = torch.arange(1, seq_len + 1).to(device)
            out = out.repeat_interleave(self.n_modalities).unsqueeze(0).repeat(batch_size, 1)
            out = self.t_encoder(out) + self._combine_modality(batch)
        out = self.transformer_encoder(out, src_key_padding_mask=attn_mask)
        return out
    
    @staticmethod
    def _combine_modality(x):
        return x.transpose(1, 2).reshape(x.size(0), x.size(1) * x.size(2), -1).squeeze()
        