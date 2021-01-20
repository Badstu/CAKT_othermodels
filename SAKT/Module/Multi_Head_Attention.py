import torch.nn as nn

from Module.ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(self.dim_per_head * num_heads, model_dim)
        self.dropout = nn.Dropout(dropout)
        # 做multi-head attention之后的layer normalization
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attn_mask=None):
        # residual 残差连接
        residual = q

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = k.size(0)
        dk = k.size(-1)

        # 线性映射
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # 分割为head_num份
        q = q.contiguous().view(batch_size * num_heads, -1, dim_per_head)
        k = k.contiguous().view(batch_size * num_heads, -1, dim_per_head)
        v = v.contiguous().view(batch_size * num_heads, -1, dim_per_head)

        # if attn_mask:
        #     attn_mask = attn_mask.repeat(num_heads, 1, 1)  # 沿着第一个维度拓展num_heads倍

        scale = dk ** -0.5

        context, attention = self.dot_product_attention(q, k, v, scale, attn_mask)

        # concat q k v
        context = context.contiguous().view(batch_size, -1, dim_per_head * num_heads)

        # final project
        output = self.linear_final(context)
        output = self.dropout(output)

        # layer normalization
        output = self.layer_norm(residual + output)

        return output, attention
