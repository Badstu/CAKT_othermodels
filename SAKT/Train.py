import os
import time
import argparse
from Trans_Model import Transformer
import numpy as np
import copy
import csv
import torch


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='assist2009_updated', type=str)
parser.add_argument('--train_data_path', default="data/assist2009_updated/assist2009_updated_train1.csv", type=str)
parser.add_argument('--test_data_path', default="data/assist2009_updated/assist2009_updated_test1.csv", type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--hidden_units', default=200, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--num_skills', default=50, type=int)
parser.add_argument('--num_steps', default=50, type=int)
parser.add_argument('--pos', default=False, type=bool)

args = parser.parse_args()
model_name = "/home/pande103/2016-EDM-master/DKT"
args.train_data_path = "./data/" + args.dataset + "/" + args.dataset + "_train.csv"
args.test_data_path = "./data/" + args.dataset + "/" + args.dataset + "_test.csv"

def read_data_from_csv_file(fileName_train, fileName_test, max_num_problems):
    inputs = []
    targets = []
    rows = []
    max_skill_num = 0
    tuple_rows = []
    train_rows = []
    test_rows = []
    with open(fileName_train, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    i = 0

    n = int(len(rows))
    print(n)
    with open(fileName_test, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print(row)
            rows.append(row)
    index = 0
    while index < len(rows) - 1:
        problems_num = int(len(rows[index + 1]))

        if problems_num <= 2:
            index += 3
            continue

        tmp_max_skill = max(map(int, rows[index + 1]))
        if tmp_max_skill > max_skill_num:
            max_skill_num = tmp_max_skill

        if problems_num < max_num_problems:
            problems = np.zeros(max_num_problems)
            correct = np.zeros(max_num_problems)
            for j in range(problems_num):
                problems[max_num_problems - j - 1] = problems[max_num_problems - j - 1] + int(
                    rows[index + 1][problems_num - j - 1])
                correct[max_num_problems - j - 1] = correct[max_num_problems - j - 1] + int(
                    rows[index + 2][problems_num - j - 1])
                tup = (problems_num, problems, correct)
            tuple_rows.append(tup)
        # #
        else:
            start_idx = 0
            while max_num_problems + start_idx <= problems_num:
                tup = (max_num_problems, [int(i) for i in rows[index + 1][start_idx:max_num_problems + start_idx]],
                       [int(i) for i in rows[index + 2][start_idx:max_num_problems + start_idx]])
                start_idx += max_num_problems
                tuple_rows.append(tup)
            # test_rows.append((rows[index], rows[index+1][-max_num_problems:], rows[index+2][-max_num_problems:]))

        index += 3
        if index == n:
            train_rows = copy.deepcopy(tuple_rows)
            tuple_rows = []

    test_rows = copy.deepcopy(tuple_rows)
    # print "The number of students is ", len(test_rows)
    # print "The number of train students is ", len(train_rows)
    # print "Finish reading data"
    return train_rows, test_rows, max_num_problems, max_skill_num + 1


train_students, test_students, max_num_problems, max_skill_num = read_data_from_csv_file(args.train_data_path,
                                                                                         args.test_data_path,
                                                                                         args.num_steps)
# args.train_num_steps = train_max_num_problems
args.num_skills = max_skill_num

weights = []
prob_arr = []
T = 0.0
t0 = time.time()
label_actual_arr = []

"""Runs the model on the given data."""
start_time = time.time()

index = 0

label_actual = []
label_pred = []
pred_labels = []
actual_labels = []
num_problems = []
model = Transformer(500, 50, 49, 8)
while (index + args.batch_size < len(train_students)):

    x = np.zeros((args.batch_size, args.num_steps - 1))
    problems = np.zeros((args.batch_size, args.num_steps - 1))
    target_id = []
    target_correctness = []
    count = 0

    for i in range(args.batch_size):
        student = train_students[index + i]
        problem_ids = student[1]
        correctness = student[2]

        for j in range(max_num_problems - 1):
            problem_id = int(problem_ids[j])
            label_index = 0
            if (int(correctness[j]) == 0):
                label_index = problem_id
            else:
                label_index = problem_id + args.num_skills

            x[i, j] = label_index

            problems[i, j] = problem_ids[j + 1]
            target_id.append(
                i * (args.num_steps - 1) * args.num_skills + j * args.num_skills + int(problem_ids[j + 1]))
            target_correctness.append(int(correctness[j + 1]))
            actual_labels.append(int(correctness[j + 1]))

    index += args.batch_size
    x = torch.LongTensor(x)
    input_len = torch.LongTensor([max_num_problems - 1]).unsqueeze(0).expand(args.batch_size, 1).squeeze(1)
    a = model(x, input_len)
    print(a)
