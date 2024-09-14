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

class FeedForward(nn.Module):
    # d_ff 中间层的维度
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class NormLayer(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super.__init__()
        self.d_model = d_model
        # 可学习
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha*((x-x.mean(dim=-1, keepdim=True)) / x.std(dim=-1,keepdim=True)+self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1) -> None:
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.att = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = FeedForward(d_model=d_model)
    def forward(self, x, mask):
        x_nrom = self.norm_1(x)
        x = self.dropout_1(self.att(x_nrom, x_nrom, x_nrom, mask))
        x_norm = self.norm_2(x)
        x = self.dropout_2(self.ffn(x_norm))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, num_layers, dropout) -> None:
        super().__init__()
        self.pe = PositionEncoder(d_model)
        self.embeded = nn.Embedding(vocab_size, d_model)
        self.norm = NormLayer(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model, heads=heads, dropout=dropout) for _ in range(num_layers)])
    def forward(self, src, mask):
        x = self.embeded(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1) -> None:
        super().__init__()
        self.norm_1 = NormLayer(d_model)
        self.norm_2 = NormLayer(d_model)
        self.norm_3 = NormLayer(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        self.att_1 = MultiHeadAttention(heads=heads, d_model=d_model, dropout=dropout)
        self.att_2 = MultiHeadAttention(heads=heads, d_model=d_model, dropout=dropout)
        self.fnn = FeedForward(d_model=d_model)
    def forward(self, x, encoder_out, src_mask, trg_mask):
        norm_x = self.norm_1(x)
        x = norm_x + self.dropout_1(self.att_1(norm_x, norm_x, norm_x, trg_mask))
        norm_x = self.norm2(x)
        x = norm_x + self.dropout_2(self.att_2(norm_x, encoder_out, encoder_out, src_mask))
        norm_x = self.norm_3(x)
        x = norm_x + self.fnn(norm_x)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, heads, num_layers, dropout) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionEncoder(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout=dropout) for _ in range(num_layers)])
        self.norm = NormLayer(d_model)
    def formward(self, trg, encoder_out, src_mask, trg_mask):
        embedding = self.embed(trg)
        x = self.pe(embedding)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, trg_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, num_layers, heads, dropout) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, heads, num_layers, dropout)
        self.decoder = Decoder(trg_vocab, d_model, heads, num_layers, dropout)
        # y映射到若干词汇
        self.linear = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        output = self.linear(decoder_out)
        return F.softmax(output)