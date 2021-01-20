from Module.Multi_Head_Attention import *
from Module.Position_Wise_Feed_Forward import *
from Module.PositionalEncoding import *
from utils.mask_util import *


# single layer
class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.ffn = FFN(model_dim, ffn_dim, dropout)

    def forward(self, x, attn_mask=None):
        # self attention
        context, attention = self.attention(x, x, x, attn_mask)

        output = self.ffn(context)

        return output


class Encoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_sequence_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.seq_embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = PositionalEncoding(model_dim, max_sequence_len, False)

    def forward(self, input_, input_lens):
        output = self.seq_embedding(input_)
        output += self.pos_embedding(input_lens).float()

        # self_attention_mask = padding_mask(input_, input_)

        attentions = []

        for encoder in self.encoder_layers:
            output = encoder(output, input_)
            # attentions.append(attention)

        return output, attentions
