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
        
# Model Experimentation: https://www.kaggle.com/lixinchu/code

################### RNAModel2: Transformer Inspired Encoder Decoder ###################

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class RNAModel2(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
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
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=depth)
        
        decoder_layers = TransformerDecoderLayer(d_model=dim, 
                                                nhead=dim//head_size, 
                                                dim_feedforward=4*dim,
                                                dropout=0.1)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layers, num_layers=depth)

        self.output_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2)

    def forward(self, x, padding_mask):
        Lmax = padding_mask.sum(-1).max()
        m = padding_mask[:, :Lmax]
        x = x[:, :Lmax]

        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_encoder(pos)
        x = self.emb(x)
        x = x + pos

        x = self.input_layer_norm(x)

        enc_output = self.transformer_encoder(x, src_key_padding_mask=~m)
        dec_output = self.transformer_decoder(x, memory=enc_output, tgt_key_padding_mask=~m.transpose(0,1))

        x = self.output_norm(dec_output)
        x = self.linear(x)
        return x

################### RNAModel3: T5 Inspired Encoder Decoder ###################

class SimplifiedLayerNorm(nn.Module):
    def __init__(self, features, epsilon=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.weight * (x - mean) / (std + self.epsilon)
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, 4 * dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * dim, dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class T5Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, num_heads=6)
        self.feed_forward = FeedForwardLayer(dim)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = SimplifiedLayerNorm(dim)
        self.norm2 = SimplifiedLayerNorm(dim)

    def forward(self, x, mask):        
        # Self-attention with layer normalization
        x_normalized_1 = self.norm1(x)
        attn_output, _ = self.self_attention(x_normalized_1, x_normalized_1, x_normalized_1, attn_mask=None)
        x = x + self.dropout(attn_output) # Residual connection 

        # Feed-forward with layer normalization
        x_normalized_2 = self.norm2(x)
        feed_forward_output = self.feed_forward(x_normalized_2)
        x = x + self.dropout(feed_forward_output)  # Residual connection
        return x

class T5CrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(dim, num_heads=6)
        self.feed_forward = FeedForwardLayer(dim)
        self.dropout = nn.Dropout(0.1)
        self.norm1 = SimplifiedLayerNorm(dim)
        self.norm2 = SimplifiedLayerNorm(dim)

    def forward(self, x, encoder_output, tgt_mask):
        # Cross-attention with layer normalization
        x_normalized_1 = self.norm1(x)
        attn_output, _ = self.cross_attention(x_normalized_1, encoder_output, encoder_output, attn_mask=None)
        x = x + self.dropout(attn_output) # Residual connection 

        # Feed-forward with layer normalization
        x_normalized_2 = self.norm2(x)
        feed_forward_output = self.feed_forward(x_normalized_2)
        x = x + self.dropout(feed_forward_output)  # Residual connection

        return x
    
class T5Encoder(nn.Module):
    def __init__(self, dim=192, depth=12):
        super().__init__()
        self.dim = dim
        self.blocks = nn.ModuleList([T5Block(dim) for _ in range(depth)])
        
    def forward(self, x, src_mask):
        for block in self.blocks:
            x = block(x, src_mask)
        return x

class T5Decoder(nn.Module):
    def __init__(self, dim=192, depth=12):
        super().__init__()
        self.dim = dim
        self.self_blocks = nn.ModuleList([T5Block(dim) for _ in range(depth)])
        self.cross_blocks = nn.ModuleList([T5CrossAttentionBlock(dim) for _ in range(depth)])  # Additional cross-attention blocks
        
    def forward(self, x, encoder_output, tgt_mask):
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            x = self_block(x, tgt_mask)
            x = cross_block(x, encoder_output, tgt_mask)  # Applying cross-attention
        return x

class RNAModel3(nn.Module): # Encoder Only
    def __init__(self, dim=192, depth=12):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=4, embedding_dim=dim, padding_idx=0)        
        self.positional_embedding = SinusoidalPosEmbedding(embed_dim=dim)
        self.encoder = T5Encoder(dim=dim, depth=depth)
        self.decoder = T5Decoder(dim=dim, depth=depth)
        self.linear = nn.Linear(dim, 2)
        
    def forward(self, x, padding_mask):
        Lmax = padding_mask.sum(-1).max()
        m = padding_mask[:, :Lmax]
        x = x[:, :Lmax]

        emb_x = self.embedding(x.long())
        pos_emb = self.positional_embedding(x)
        x = emb_x + pos_emb
        
        encoder_output = self.encoder(x, src_mask=~m)
        decoder_output = self.decoder(x, encoder_output,tgt_mask=~m.transpose(0,1))
        output = self.linear(decoder_output)
        output = F.log_softmax(output, dim=-1)
        return output

################### RNAModel4: BERT Inspired Encoder ###################

class EncoderLayer(nn.Module):
    def __init__(self, dim=192, n_heads=6, d_ff=4*192):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(dim, n_heads)
        self.feed_forward = PositionwiseFeedForward(dim, d_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        
    def forward(self, x, mask):
        attended = self.self_attention(x, x, x,mask)
        attended = self.norm1(attended + x)
        attended = self.feed_forward(attended)
        output = self.norm2(attended + attended)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=192, n_heads=6):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        
        self.W_Q = nn.Linear(dim, dim)
        self.W_K = nn.Linear(dim, dim)
        self.W_V = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.dim // self.n_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim // self.n_heads)**0.5
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1,self.dim)
        return self.linear(attended)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim=192, d_ff=4*192):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(dim, d_ff)
        self.linear2 = nn.Linear(d_ff, dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class RNAModel4(nn.Module):
    def __init__(self, num_embeddings=4, dim=192, depth=12, n_heads=6, d_ff=4*192):
        super(RNAModel4, self).__init__()
        self.embedding = nn.Embedding(num_embeddings,dim)
        self.positional_embedding = SinusoidalPosEmbedding(dim)
        self.layers = nn.ModuleList([EncoderLayer(dim, n_heads, d_ff) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, 2)
        
    def forward(self, x, mask):
        embedded = self.embedding(x)
        pos_embedded = self.positional_embedding(x)       
        embedded = embedded + pos_embedded

        output = embedded
        for layer in self.layers:
            output = layer(output, mask)
        
        output = self.norm(output)
        output = self.linear(output)
        return output
