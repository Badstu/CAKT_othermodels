import torch
import torch.nn as nn

from memory import DKVMN


class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim,
                 memory_size, memory_key_state_dim, memory_value_state_dim, final_fc_dim, student_num=None):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num

        self.input_embed_linear = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)
        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.q_embed_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)
        self.mem = DKVMN(memory_size=self.memory_size,
                   memory_key_state_dim=self.memory_key_state_dim,
                   memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        # 题目序号从1开始
        # nn.embedding输入是一个下标的列标，输出是对应的嵌入
        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        # nn.init.constant(self.input_embed_linear.bias, 0)
        # nn.init.normal(self.input_embed_linear.weight, std=0.02)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_data, qa_data, target, student_id=None):

        batch_size = q_data.shape[0]   #32
        seqlen = q_data.shape[1]   #200

        ## qt && (q,a) embedding
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        ## copy mk batch times for dkvmn
        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        ## slice data for seqlen times by axis 1
        # torch.chunk(tensor, chunk_num, dim)
        slice_q_data = torch.chunk(q_data, seqlen, 1)
        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)


        value_read_content_l = []
        input_embed_l = []

        for i in range(seqlen):
            ## Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            ## Read Process
            read_content = self.mem.read(correlation_weight)

            ## save intermedium data
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            ## Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            new_memory_value = self.mem.write(correlation_weight, qa)


        # Projection
        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        ## Project rt
        input_embed_content = input_embed_content.view(batch_size * seqlen, -1)
        input_embed_content = torch.tanh(self.input_embed_linear(input_embed_content))
        input_embed_content = input_embed_content.view(batch_size, seqlen, -1)

        ## Concat Read_Content and input_embedding_value
        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size*seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        # predicts = torch.cat([predict_logs[i] for i in range(seqlen)], 1)
        target_1d = target                   # [batch_size * seq_len, 1]
        # mask = target_1d.ge(0)               # [batch_size * seq_len, 1]
        mask = q_data.gt(0).view(-1, 1)
        # pred_1d = predicts.view(-1, 1)           # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)           # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target