import numpy as np
import math

class DATA(object):
    def __init__(self, n_question, seqlen, separate_char, name="data"):
        # In the ASSISTments2009 dataset:
        # param: n_queation = 110
        #        seqlen = 200
        self.separate_char = separate_char
        self.n_question = n_question
        """
        self.seqlen = seqlen+1
        """
        self.seqlen = seqlen

    ### data format
    ### 15
    ### 1,1,1,1,7,7,9,10,10,10,10,11,11,45,54
    ### 0,1,1,1,1,1,0,0,1,1,1,1,1,0,0
    def load_data(self, path):
        f_data = open(path, 'r')
        q_a_data = []
        q_target_data = []
        answer_data = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]

                # # start split the data
                # n_split = 1
                # ## 多取一个题目
                # new_seq_len = self.seqlen + 1
                # if len(Q) > new_seq_len:
                #     n_split = math.floor(len(Q) / new_seq_len)
                #     if len(Q) % new_seq_len:
                #         n_split = n_split + 1
                # for k in range(n_split):
                #     question_sequence = []
                #     answer_sequence = []
                #     if k == n_split - 1:
                #         end_index = len(A)
                #     else:
                #         end_index = (k + 1) * new_seq_len
                question_sequence = []
                answer_sequence = []
                for i in range(len(Q)):
                    if len(Q[i]) > 0:
                        # int(A[i]) is in {0,1}
                        x_index = int(Q[i]) + int(A[i]) * self.n_question
                        question_sequence.append(int(Q[i]))
                        answer_sequence.append(x_index)
                    else:
                        print(Q[i])
                # print('instance:-->', len(instance),instance)
                if len(question_sequence) > 1:
                    q_a_sequence = answer_sequence[:-1]
                    q_target_sequence = question_sequence[1:]
                    answer_sequence = answer_sequence[1:]

                    q_a_data.append(q_a_sequence)
                    q_target_data.append(q_target_sequence)
                    answer_data.append(answer_sequence)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        # q_a_dataArray = np.zeros((len(q_a_data), self.seqlen))
        # for j in range(len(q_a_data)):
        #     dat = q_a_data[j]
        #     q_a_dataArray[j, :len(dat)] = dat

        # q_target_dataArray = np.zeros((len(q_target_data), self.seqlen))
        # for j in range(len(q_target_data)):
        #     q_target_dat = q_target_data[j]
        #     q_target_dataArray[j, :len(q_target_dat)] = q_target_dat

        # answer_dataArray = np.zeros((len(answer_data), self.seqlen))
        # for j in range(len(answer_data)):
        #     answer_dat = answer_data[j]
        #     answer_dataArray[j, :len(answer_dat)] = answer_dat

        # return q_a_dataArray, q_target_dataArray, answer_dataArray

        return q_a_data, q_target_data, answer_data
