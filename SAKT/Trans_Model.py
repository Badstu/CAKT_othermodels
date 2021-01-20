from Module.Encoder import *


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 num_skill,
                 src_max_len,
                 num_heads=8,
                 model_dim=200,
                 num_layers=1,
                 ffn_dim=2048,
                 dropout=0.2):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, num_skill, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src_seq, src_len):
        output, src_self_attention = self.encoder(src_seq, src_len)

        output = self.linear(output)
        output = self.sigmoid(output)

        return output
