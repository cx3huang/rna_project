import math
import pandas as pd
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

#torches default attention mechanism doesn't support rpe
def custom_attention(query, key, value, rel_pos_embedding, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Adding relative positional scores
    rel_pos_scores = torch.matmul(query, rel_pos_embedding.transpose(-2, -1))
    scores += rel_pos_scores

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class RelativePositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.rel_pos_embedding = nn.Parameter(torch.randn(max_len * 2 + 1, d_model))

        
    def forward(self, q_lens, k_lens):
        max_len = self.rel_pos_embedding.shape[0] // 2
        batch_size = q_lens.size(0)

        rel_pos_embeddings = []
        for i in range(batch_size):
            q_len = q_lens[i].item()
            k_len = k_lens[i].item()
            rel_indices = torch.arange(-min(k_len, max_len), min(q_len, max_len), device=self.rel_pos_embedding.device)
            rel_indices += max_len  # shift to positive indices
            rel_pos_embeddings.append(self.rel_pos_embedding[rel_indices])

        # Pad the embeddings to the same size
        padded_rel_pos_embeddings = nn.utils.rnn.pad_sequence(rel_pos_embeddings, batch_first=True)
        return padded_rel_pos_embeddings


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
    def __init__(self, dim=192, depth=12, head_size=32, max_len=206):
        super().__init__()
        self.d_model = dim

        self.emb = nn.Embedding(num_embeddings=4, embedding_dim=dim, padding_idx=0)
        # self.pos_encoder = SinusoidalPosEmbedding(embed_dim=dim)
        self.relative_pos_emb = RelativePositionalEmbedding(max_len, dim)
        self.input_layer_norm = nn.LayerNorm(dim)

        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, 
                                                    nhead=dim//head_size, 
                                                    dim_feedforward=4*dim,
                                                    dropout=0.1,
                                                    activation=nn.GELU(),
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=depth)
        
        self.output_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2)

    def forward(self, x, padding_mask):
        # Lmax = padding_mask.sum(-1).max()
        # # Lmax = 206
        # m = padding_mask[:,:Lmax]
        # x = x[:,:Lmax]

        # pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        # pos = self.pos_encoder(pos)
        # x = self.emb(x)
        # x = x + pos

        batch_size, seq_len = x.shape[:2]
        q_lens = padding_mask.sum(dim=1)
        k_lens = q_lens.clone()

        rel_pos_emb = self.relative_pos_emb(q_lens, k_lens)

        x = self.emb(x)  # Token embeddings
        x += rel_pos_emb[:, :seq_len, :]  # Align and add positional embeddings

        x = self.input_layer_norm(x)

        x = self.transformer_encoder(x, src_key_padding_mask=~padding_mask)
        
        x = self.output_norm(x)
        x = self.linear(x)
        # x = x.squeeze()
        return x


