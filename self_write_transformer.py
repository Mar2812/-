import torch
import torch.nn
import math

class PositionEncoder(torch.nn.Module):
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

