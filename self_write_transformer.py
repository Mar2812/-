import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionEncoder(nn.Module):
    def __init__(self, d_model, max_len=80) -> None:
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / math.pow(10000, 2 * i / d_model))
                pe[pos, i+1] = math.cos(pos / math.pow(10000, 2 * (i + 1) / d_model))
        # 这是由于实际中 x 的形状通常是 (batch_size, seq_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        # 第一个维度是batch_size，由于上面unsqueeze了，那么原来的第一个维度（序列长度）就是第二个维度了
        x = x + self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.d_head = d_model / heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        # 得出的scores为batch_size, heads, q_seq_len, k_seq_len, 所以要归一化q对key的注意力
        if mask:
            # 这是由于mask可能是batchsize, seq_len，而此处为多头，维度对其
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0, -1e9)
        scores = F.softmax(scores, dim=-1)
        if dropout:
            scores = dropout(scores)
        output = torch.matmul(scores, v)
        return output
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape()[0]
        # 得出q,k,v为batch_size, heads, seq_len, d_head
        q = self.q_linear(q).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
        v = self.k_linear(v).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
        scores = self.attention(q, k, v, mask, self.dropout)
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)
        return output
        
