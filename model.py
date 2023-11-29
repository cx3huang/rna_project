import math
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


# class LearnedPosEmbedding(nn.Embedding):
#     def __init__(self, num_embeddings, embedding_dim, padding_idx):
#         if padding_idx is not None:
#             num_embeddings_ = num_embeddings + padding_idx + 1
#         else:
#             num_embeddings_ = num_embeddings
#         super().__init__(num_embeddings_, embedding_dim, padding_idx)
#         self.max_positions = num_embeddings
    
#     def forward(self, input: Tensor):
#         mask = input.ne(self.padding_idx).int()
#         positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
#         return F.embedding(
#             positions,
#             self.weight,
#             self.padding_idx,
#             self.max_norm,
#             self.norm_type,
#             self.scale_grad_by_freq,
#             self.sparse,
#         )


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, embed_dim=16, M=10000, padding_idx=None, weights=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M

    def forward(self, x):
        return self.get_embedding(x)

    def get_embedding(self, x: Tensor):
        half_dim = self.embed_dim // 2
        emb = math.log(self.M) / (half_dim)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=x.device) * (-emb))
        emb = x[..., None] * emb[None, ...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RNAModel(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs) -> None:
        super().__init__()
        self.d_model = dim

        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=dim, padding_idx=0)
        self.pos_encoder = SinusoidalPosEmbedding(embed_dim=dim)
        self.input_layer_norm = nn.LayerNorm(dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=dim//head_size, 
                                                    dim_feedforward=4*dim,
                                                    dropout=0.1,
                                                    activation=nn.GELU(),
                                                    batch_first=True,)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=depth)
        
        self.output_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2)

    def forward(self, x, padding_mask):
        Lmax = padding_mask.sum(-1).max()
        m = padding_mask[:,:Lmax]
        x = x[:,:Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_encoder(pos)
        x = self.emb(x)
        x = x + pos

        x = self.input_layer_norm(x)

        x = self.transformer_encoder(x, src_key_padding_mask=~m)
        
        x = self.output_norm(x)
        x = self.linear(x)
        # x = x.squeeze()
        return x
        

