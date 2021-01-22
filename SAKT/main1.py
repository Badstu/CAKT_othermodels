import os
import time
import argparse
import tensorflow as tf
# from sampler import WarpSampler
from model import Model
import numpy as np
import sys
import copy
import random
import csv
from math import sqrt
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


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
    # print(n)
    with open(fileName_test, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            # print(row)
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

            if problems_num - start_idx > 0:
                problems = np.zeros(max_num_problems)
                correct = np.zeros(max_num_problems)
                problems[max_num_problems - (problems_num - start_idx):] = rows[index + 1][start_idx: problems_num]
                correct[max_num_problems - (problems_num - start_idx):] = rows[index + 2][start_idx: problems_num]
                tup = (max_num_problems, [int(i) for i in problems], [int(i) for i in correct])
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


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='synthetic', type=str)
parser.add_argument('--train_set', type=int, default=1)
parser.add_argument('--num_heads', default=2, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--hidden_units', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--num_blocks', default=1, type=int)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--num_skills', default=50, type=int)
parser.add_argument('--num_steps', default=200, type=int)
parser.add_argument('--pos', default=False, type=bool)
parser.add_argument('--first_k', default=8, type=int)

args = parser.parse_args()
# model_name = "/home/pande103/2016-EDM-master/DKT"
if args.dataset == 'assist2017':
    args.train_data_path = "../dataset/" + args.dataset + "/train_valid_test/assist2017_train" + str(args.train_set) + ".csv"
    args.valid_data_path = "../dataset/" + args.dataset + "/train_valid_test/assist2017_valid" + str(args.train_set) + ".csv"
    args.test_data_path = "../dataset/" + args.dataset + "/train_valid_test/assist2017_test.csv"
elif args.dataset == 'synthetic':
    args.train_data_path = "../dataset/synthetic/naive_c5_q50_s4000_v0_train" + str(args.train_set) + ".csv"
    args.valid_data_path = "../dataset/synthetic/naive_c5_q50_s4000_v0_valid" + str(args.train_set) + ".csv"
    args.test_data_path = "../dataset/synthetic/naive_c5_q50_s4000_v0_test.csv"
else:
    args.train_data_path = "../dataset/" + args.dataset + "/" + args.dataset + "_train" + str(args.train_set) + ".csv"
    args.valid_data_path = "../dataset/" + args.dataset + "/" + args.dataset + "_valid" + str(args.train_set) + ".csv"
    args.test_data_path = "../dataset/" + args.dataset + "/" + args.dataset + "_test.csv"

if args.dataset in ['assist2017', 'STATICS']:
    args.batch_size = 32

print(args)
# print(args.train_data_path)
# print(args.valid_data_path)
# print(args.test_data_path)

train_students, valid_students, max_num_problems, max_skill_num = read_data_from_csv_file(
    args.train_data_path, args.valid_data_path, args.num_steps)
_, test_students, _, max_sn = read_data_from_csv_file(
    args.test_data_path, args.test_data_path, args.num_steps)
max_skill_num = max(max_skill_num, max_sn)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.allow_soft_placement = True
# args.train_num_steps = train_max_num_problems
args.num_skills = max_skill_num
sess = tf.Session(config=config)

# sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
# training model
with tf.variable_scope("model", reuse=None):
    m = Model(True, args)
# testing model
with tf.variable_scope("model", reuse=True):
    mtest = Model(False, args, reuse=True)  # testing model
# mtest = Model(test_students, False, args)
sess.run(tf.global_variables_initializer())
weights = []
prob_arr = []
T = 0.0
t0 = time.time()
label_actual_arr = []
# frec = open('record.csv', 'w')


def run_epoch(session, m, students, epoch, eval_op, verbose=False, is_training=True):
    """Runs the model on the given data."""
    start_time = time.time()

    index = 0

    label_actual = []
    label_pred = []
    num_problems = []

    while (index + args.batch_size <= len(students)):

        pred_labels = []
        actual_labels = []

        x = np.zeros((args.batch_size, args.num_steps - 1))
        problems = np.zeros((args.batch_size, args.num_steps - 1))
        target_id = []
        target_correctness = []
        count = 0

        for i in range(args.batch_size):
            student = students[index + i]
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

        pred, _ = session.run([m._pred, eval_op], feed_dict={
            m._input_data: x, m.target_id: target_id,
            m.target_correctness: target_correctness, m.is_training: is_training, m.problems: problems})

        for p in pred:
            pred_labels.append(p)

        # modify
        x = x.reshape([-1])
        x_mask = np.array(np.nonzero(x))
        x_mask = x_mask.reshape([-1])
        actual_labels = np.array(actual_labels)
        pred_labels = np.array(pred_labels)
        # modify
        for term in x_mask:
            label_actual.append(actual_labels[term])
            label_pred.append(pred_labels[term])

    rmse = sqrt(mean_squared_error(label_actual, label_pred))
    fpr, tpr, thresholds = metrics.roc_curve(label_actual, label_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return rmse, auc


num_batch = len(train_students) / args.batch_size
# f_log = open("auc" + "_" + args.dataset + "_" + str(args.hidden_units) + "_" + str(args.num_heads) + "_" + str(
#     args.num_steps) + "_" + str(args.num_blocks) + ".csv", 'w+')
best_valid_auc = 0
correspond_train_auc = 0
correspond_test_auc = 0
for epoch in range(1, args.num_epochs + 1):
    rmse, auc = run_epoch(sess, m, train_students, epoch, m.train_op, verbose=True, is_training=True)

    if epoch % 5 == 0:
        # st = str(auc)
        # st2 = str(rmse)

        print("Epoch: %d Train Metrics:\n  train_auc: %.5f" % (epoch + 1, auc))
        
        valid_rmse, valid_auc = run_epoch(sess, mtest, valid_students, epoch, tf.no_op(), is_training=False)
        print("Epoch: %d Valid Metrics:\n  valid_auc: %.5f" % (epoch + 1, valid_auc))
        test_rmse, test_auc = run_epoch(sess, mtest, test_students, epoch, tf.no_op(), is_training=False)
        print("Epoch: %d Test Metrics:\n  test_auc: %.5f" % (epoch + 1, test_auc))

        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            correspond_train_auc = auc
            correspond_test_auc = test_auc

print("DATASET: {}, NUM_HEADS: {}, HIDDEN_UNITS: {}, LR: {}".format(args.dataset, args.num_heads, args.hidden_units, args.lr))
print("BEST VALID AUC: {}, CORRESPOND TRAIN AUC: {}, CORRESPOND TEST AUC: {}".format(best_valid_auc, correspond_train_auc, correspond_test_auc))
