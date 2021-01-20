import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.soft_max = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # 对每个q 计算Q和V的依赖权重并对V加权求和
        # attention = soft_max(Q*K.T)/scale*V
        # Q [B,S1,E]
        # K = V [B,S2,E]
        # self-attention Q=K=V
        attention = torch.bmm(q, k.transpose(1, 2))  # 第一维度和第二维度转置
        if scale:
            attention = attention * scale
        # if attn_mask:  # 做mask
        #     attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算soft_max
        attention = self.soft_max(attention)
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        return context, attention
