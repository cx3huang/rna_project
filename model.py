import math
import pandas as pd
import numpy as np
import torch
import torch.functional as F
from torch import Tensor
import torch.nn as nn
from dataset import RNADataLoader, RNADataset

class LearnedPosEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        if padding_idx is not None:
            num_embeddings_ = num_embeddings + padding_idx + 1
        else:
            num_embeddings_ = num_embeddings
        super().__init__(num_embeddings_, embedding_dim, padding_idx)
        self.max_positions = num_embeddings
    
    def forward(self, input: Tensor):
        mask = input.ne(self.padding_idx).int()
        positions = (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx
        return F.embedding(
            positions,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

"""TODO: Research these embeddings"""
class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, embed_dim=16, M=10000, padding_idx=None, weights=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.M = M

    def forward(self, x):
        return self.get_embedding(x)

    def get_embedding(self, x: Tensor):
        half_dim = self.embed_dim // 2
        emb = math.log(self.M) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=x.device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class RNAModel(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs) -> None:
        super.__init__()
        self.emb = nn.Embedding(4, dim)
        self.pos_encoder = SinusoidalPosEmbedding(dim)
        