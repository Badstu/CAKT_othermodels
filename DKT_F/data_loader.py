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
        repeated_time_gap = []
        past_trail_counts = []
        for lineID, line in enumerate(f_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 3 == 1:
                Q = line.split(self.separate_char)
                if len(Q[len(Q) - 1]) == 0:
                    Q = Q[:-1]
                # print(len(Q))
            elif lineID % 3 == 2:
                A = line.split(self.separate_char)
                if len(A[len(A) - 1]) == 0:
                    A = A[:-1]
                # print(len(A),A)

                # start split the data
                n_split = 1
                # print('len(Q):',len(Q))
                ## 多取一个题目
                new_seq_len = self.seqlen + 1
                if len(Q) > new_seq_len:
                    n_split = math.floor(len(Q) / new_seq_len)
                    if len(Q) % new_seq_len:
                        n_split = n_split + 1
                # print('n_split:',n_split)
                for k in range(n_split):
                    question_sequence = []
                    answer_sequence = []
                    if k == n_split - 1:
                        end_index = len(A)
                    else:
                        end_index = (k + 1) * new_seq_len
                    for i in range(k * new_seq_len, end_index):
                        if len(Q[i]) > 0:
                            # int(A[i]) is in {0,1}
                            x_index = int(Q[i]) + int(A[i]) * self.n_question
                            question_sequence.append(int(Q[i]))
                            answer_sequence.append(x_index)
                        else:
                            print(Q[i])
                    # print('instance:-->', len(instance),instance)
                    sub_rtg = [8]
                    sub_ptc = []
                    rtg = [0] * (self.n_question + 1)
                    ptc = [0] * (self.n_question + 1)
                    for i in range(len(question_sequence)):
                        sub_ptc.append(math.floor(np.log2(ptc[question_sequence[i]])) if ptc[question_sequence[i]] != 0 else 0)
                        ptc[question_sequence[i]] += 1
                        j = i - 1
                        while j >= 0:
                            if question_sequence[j] == question_sequence[i]:
                                rtg[question_sequence[i]] = i - j
                                sub_rtg.append(math.floor(np.log2(rtg[question_sequence[i]])))
                                break
                            if j == 0:
                                sub_rtg.append(8)
                            j -= 1

                    if len(question_sequence) > 1:
                        q_a_sequence = answer_sequence[:-1]
                        q_target_sequence = question_sequence[1:]
                        answer_sequence = answer_sequence[1:]

                        q_a_data.append(q_a_sequence)
                        q_target_data.append(q_target_sequence)
                        answer_data.append(answer_sequence)
                        repeated_time_gap.append(sub_rtg)
                        past_trail_counts.append(sub_ptc)

        f_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        ### convert data into ndarrays for better speed during training
        q_a_dataArray = np.zeros((len(q_a_data), self.seqlen))
        for j in range(len(q_a_data)):
            dat = q_a_data[j]
            q_a_dataArray[j, :len(dat)] = dat

        q_target_dataArray = np.zeros((len(q_target_data), self.seqlen))
        for j in range(len(q_target_data)):
            q_target_dat = q_target_data[j]
            q_target_dataArray[j, :len(q_target_dat)] = q_target_dat

        answer_dataArray = np.zeros((len(answer_data), self.seqlen))
        for j in range(len(answer_data)):
            answer_dat = answer_data[j]
            answer_dataArray[j, :len(answer_dat)] = answer_dat

        repeated_time_gap_dataArray = np.zeros((len(repeated_time_gap), self.seqlen + 1, 9))
        for i in range(len(repeated_time_gap)):
            for j in range(len(repeated_time_gap[i])):
                repeated_time_gap_dataArray[i, j, repeated_time_gap[i][j]] = 1

        past_trail_counts_dataArray = np.zeros((len(past_trail_counts), self.seqlen + 1, 9))
        for i in range(len(past_trail_counts)):
            for j in range(len(past_trail_counts[i])):
                past_trail_counts_dataArray[i, j, past_trail_counts[i][j]] = 1

        seq_time_gap_dataArray = np.ones((len(repeated_time_gap), self.seqlen + 1, 1))

        # dataArray: [ array([[],[],..])] Shape: (3633, 200)
        return q_a_dataArray, q_target_dataArray, answer_dataArray, repeated_time_gap_dataArray, past_trail_counts_dataArray, seq_time_gap_dataArray

