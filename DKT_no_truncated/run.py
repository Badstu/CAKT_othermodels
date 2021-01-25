import math

import numpy as np
import torch
from sklearn import metrics
from torch import nn

import utils as utils

def train(model, params, optimizer, q_a_data, q_target_data, answer_data):
    N = int(math.floor(len(q_a_data) / params.batch_size))

    # shuffle_index = np.random.permutation(len(q_a_data))
    # q_a_data = q_a_data[shuffle_index]
    # q_target_data = q_target_data[shuffle_index]
    # answer_data = answer_data[shuffle_index]

    criterion = nn.BCELoss()
    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    for idx in range(N):
        q_a_seq = q_a_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        answer_seq = answer_data[idx * params.batch_size:(idx + 1) * params.batch_size]

        max_len = max(len(q_a_seq[i]) for i in range(params.batch_size))

        q_a_dataArray = np.zeros((params.batch_size, max_len))
        q_target_dataArray = np.zeros((params.batch_size, max_len))
        answer_dataArray = np.zeros((params.batch_size, max_len))
        for j in range(params.batch_size):
            dat = q_a_seq[j]
            q_a_dataArray[j, :len(dat)] = dat

            q_target_dat = q_target_seq[j]
            q_target_dataArray[j, :len(q_target_dat)] = q_target_dat

            answer_dat = answer_seq[j]
            answer_dataArray[j, :len(answer_dat)] = answer_dat

        # q_a_dataArray = q_a_data[idx]
        # q_target_dataArray = q_target_data[idx]
        # answer_dataArray = answer_data[idx]

        target = (answer_dataArray - 1) / params.n_question
        target = np.floor(target)
        input_q_target = utils.variable(torch.LongTensor(q_target_dataArray), params.gpu)
        input_x = utils.variable(torch.LongTensor(q_a_dataArray), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)


        model.zero_grad()
        loss, filtered_pred, filtered_target = model(input_x, input_q_target_1d, target_1d)
        # loss = criterion(filtered_pred, filtered_target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.max_grad_norm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc

def test(model, params, optimizer, q_a_data, q_target_data, answer_data):
    N = int(math.floor(len(q_a_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    for idx in range(N):
        q_a_seq = q_a_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        q_target_seq = q_target_data[idx * params.batch_size:(idx + 1) * params.batch_size]
        answer_seq = answer_data[idx * params.batch_size:(idx + 1) * params.batch_size]

        max_len = max(len(q_a_seq[i]) for i in range(params.batch_size))

        q_a_dataArray = np.zeros((params.batch_size, max_len))
        q_target_dataArray = np.zeros((params.batch_size, max_len))
        answer_dataArray = np.zeros((params.batch_size, max_len))
        for j in range(params.batch_size):
            dat = q_a_seq[j]
            q_a_dataArray[j, :len(dat)] = dat

            q_target_dat = q_target_seq[j]
            q_target_dataArray[j, :len(q_target_dat)] = q_target_dat

            answer_dat = answer_seq[j]
            answer_dataArray[j, :len(answer_dat)] = answer_dat

        # q_a_dataArray = q_a_data[idx]
        # q_target_dataArray = q_target_data[idx]
        # answer_dataArray = answer_data[idx]

        target = (answer_dataArray - 1) / params.n_question
        target = np.floor(target)
        input_q_target = utils.variable(torch.LongTensor(q_target_dataArray), params.gpu)
        input_x = utils.variable(torch.LongTensor(q_a_dataArray), params.gpu)
        target = utils.variable(torch.FloatTensor(target), params.gpu)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)
        input_q_target_to_1d = torch.chunk(input_q_target, params.batch_size, 0)
        input_q_target_1d = torch.cat([input_q_target_to_1d[i] for i in range(params.batch_size)], 1)
        input_q_target_1d = input_q_target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(input_x, input_q_target_1d, target_1d)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss / N, accuracy, auc
