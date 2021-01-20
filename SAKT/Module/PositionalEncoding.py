import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_len, zero_padding=True):
        super(PositionalEncoding, self).__init__()
        # '''
        # 初始化 d_model 模型输出维度
        #       max_sequence_len 文本序列最大长度
        # '''
        # 构造PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
            for pos in range(max_sequence_len)
        ])
        # 对所有pos按照奇偶数分别编码
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        # 位置为0的position encoding为PAD
        position_encoding = torch.tensor(position_encoding)
        if zero_padding:
            pad_row = torch.zeros([1, d_model])
            position_encoding = torch.cat((pad_row, position_encoding))

        # 嵌入操作 无需梯度更新 projection
        self.position_encoding = nn.Embedding(max_sequence_len, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_lens):
        max_sequence_len = max(input_lens)
        tensor = torch.cuda.LongTensor if input_lens.is_cuda else torch.LongTensor
        # 每个序列位置对其
        # 1 2 3 4 5 0 0 0
        # 1 2 3 0 0 0 0 0
        # 1 2 3 4 5 6 0 0
        input_pos = tensor([
            list(range(0, int(length))) + [0] * (int(max_sequence_len) - int(length)) for length in input_lens
        ])
        # tensor([
        #     list(range(1, length + 1)) + torch.Tensor([0]) * (max_sequence_len - length) for length in input_lens
        # ])
        return self.position_encoding(input_pos)
