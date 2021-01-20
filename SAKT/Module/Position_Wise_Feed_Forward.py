import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(FFN, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x size [B,L,E]
        # Convolution作用域为第一维度
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))

        # 维度还原
        output = self.dropout(output.transpose(1, 2))

        # residual
        output = self.layer_norm(x + output)
        return output
