import torch


# embedding之前填充 填充的是相互关系矩阵，q对k中为0的地方注意力为0
def padding_mask(seq_k, seq_q):
    # seq_k ,seq_q size [B,L]
    len_q = seq_q.size(1)
    # 'PAD' == 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).view(-1, len_q, -1)  # [B,L_q,L_k]
    return pad_mask


# 使得attention不能对未来产生依赖
def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask
