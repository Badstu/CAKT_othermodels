import os
import sys
import random
import torch
from torch import nn
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DataAssist2009(Dataset):
    # TODO 随机划分训练测试集
    # @params opt:['train', 'valid', 'test']
    def __init__(self, csv_pathes, opt='train', random_state=42):
        # knowlege: 1~110
        # label: 0/1
        self.opt = opt
        self.pn_list = []
        self.kms = []
        self.cms = []

        for csv_path in csv_pathes:
            pn_list, km, cm = self.get_data(csv_path)
            self.pn_list.extend(pn_list)
            self.kms.extend(km)
            self.cms.extend(cm)


        total_length = len(self.pn_list)

        self.train_pn_list, self.test_pn_list,\
            self.train_kms, self.test_kms, \
            self.train_cms, self.test_cms = train_test_split(self.pn_list, self.kms, self.cms,
                                                    test_size=0.1, random_state=random_state)

        self.train_pn_list, self.valid_pn_list, \
            self.train_kms, self.valid_kms, \
            self.train_cms, self.valid_cms = train_test_split(self.train_pn_list, self.train_kms, self.train_cms,
                                                         test_size=0.2, random_state=random_state)

        if self.opt == 'train':
            self.pn_list, self.kms, self.cms = self.train_pn_list, self.train_kms, self.train_cms
        elif self.opt == 'valid':
            self.pn_list, self.kms, self.cms = self.valid_pn_list, self.valid_kms, self.valid_cms
        elif self.opt == 'test':
            self.pn_list, self.kms, self.cms = self.test_pn_list, self.test_kms, self.test_cms

    def __len__(self):
        if self.opt == 'train':
            return len(self.train_pn_list)
        elif self.opt == 'valid':
            return len(self.valid_pn_list)
        elif self.opt == 'test':
            return len(self.test_pn_list)

    def __getitem__(self, idx):
        data_length = self.pn_list[idx]
        seq = self.kms[idx]
        label = self.cms[idx]
        return data_length, seq, label

    def get_data(self, csv_path):
        '''
        practice_number(pn): 15
        knowledge(km): 65,65,65,65,65,65,65,65,65,65,65,65,65,65,65
        correct(cm): 1,1,0,1,0,1,1,1,1,0,1,1,1,1,1
        '''

        pn_list = []
        km = []
        cm = []

        flag = True
        with open(csv_path) as f:
            for idx, line in enumerate(f.readlines(), 3):
                if idx % 3 == 0:
                    pn = int(line)
                    if pn < 5 or pn > 200:
                        flag = False
                        continue
                    else:
                        flag = True

                    pn_list.append(pn)
                if (idx - 1) % 3 == 0 and flag:
                    per_k = list(map(lambda x: int(x), line[:-1].split(',')))
                    km.append(torch.Tensor(per_k))
                if (idx - 2) % 3 == 0 and flag:
                    per_c = list(map(lambda x: int(x), line[:-1].split(',')))
                    cm.append(torch.Tensor(per_c))
        return pn_list, km, cm

