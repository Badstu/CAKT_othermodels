from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchnet import meter


def train_ekt(opt, vis, model, data_loader, epoch, lr, optimizer):
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    train_loss_list = []

    all_pred = []
    all_target = []

    flag = 0
    for ii, (batch_len, batch_seq, batch_label) in enumerate(data_loader):
    # for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(data_loader)):
        # batch_seq: (batch_size, max_batch_len)
        # batch_label: (batch_size, max_batch_len, (next_qnumber, next_0_1))
        batch_seq = batch_seq.to(opt.device)
        each_seq = batch_seq.squeeze(0)
        batch_label = batch_label.float().to(opt.device)

        if batch_len.item() <= 5:
            continue

        loss = 0
        auc = 0

        gt_label = []
        pred_label = []

        hidden_state = None
        for i, item in enumerate(each_seq):
            label = 0 if item <= opt.output_dim else 1
            label = torch.Tensor([label])
            label = label.float().to(opt.device)


            # model predict
            prediction_score, hidden_state = model(item, hidden=hidden_state)
            # print(prediction_score.item(), label.item())

            # calculate each record loss
            loss += criterion(prediction_score.view_as(label), label)
            

            # form predict vector to calc auc
            gt_label.append(label.item())
            pred_label.append(prediction_score.item())

        
        # if np.sum(gt_label) == len(gt_label) or np.sum(gt_label) == 0:
        #     continue
        print(gt_label, pred_label)
        # print(roc_auc_score(gt_label, pred_label))

        all_pred.append(pred_label)
        all_target.append(gt_label)

        if len(gt_label) == 0:
            continue

        loss /= len(gt_label)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # auc = roc_auc_score(gt_label, pred_label)
        auc_meter.add(auc)
        loss_meter.add(loss.item())

        train_loss_list.append(str(loss_meter.value()[0]))  # 训练到目前为止所有的loss平均值
        if opt.vis and (ii) % opt.plot_every_iter == 0:
            vis.plot("train_loss", loss_meter.value()[0])
            vis.plot("train_auc", auc_meter.value()[0])
            vis.log("epoch:{epoch}, lr:{lr:.5f}, train_loss:{loss:.5f}, train_auc:{auc:.5f}".format(epoch=epoch,
                                                                                                    lr=lr,
                                                                                                    loss=loss_meter.value()[0],
                                                                                                    auc=auc_meter.value()[0]))
    pred_1d = np.concatenate(all_pred, axis=0)
    target_1d = np.concatenate(all_target, axis=0)

    print(pred_1d, target_1d)
    auc = roc_auc_score(target_1d, pred_1d)
    # print('train overall auc = ', auc)

    # with open('./train_is_text=False', mode='a') as file:
    #     for i in range(len(target_1d)):
    #         file.write(str(target_1d[i]))
    #         file.write(', ')
    #     file.write('\n')
    #     for i in range(len(pred_1d)):
    #         file.write(str(np.round(pred_1d[i], decimals=8)))
    #         file.write(', ')
    #     file.write('\n')
    #     file.write(str(auc))
    #     file.write('\n')

    return loss_meter, auc, train_loss_list


@torch.no_grad()
def valid_ekt(opt, vis, model, valid_loader, epoch):
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    loss_meter = meter.AverageValueMeter()
    auc_meter = meter.AverageValueMeter()

    loss_meter.reset()
    auc_meter.reset()
    val_loss_list = []

    all_pred = []
    all_target = []

    for ii, (batch_len, batch_seq, batch_label) in enumerate(valid_loader):
    # for ii, (batch_len, batch_seq, batch_label) in tqdm(enumerate(valid_loader)):
        batch_seq = batch_seq.to(opt.device)
        each_seq = batch_seq.squeeze(0)
        batch_label = batch_label.float().to(opt.device)

        loss = 0
        auc = 0

        if batch_len.item() <= 5:
            continue

        gt_label = []
        pred_label = []

        hidden_state = None
        for i, item in enumerate(each_seq):
            label = 0 if item <= opt.output_dim else 1
            label = torch.Tensor([label])
            label = label.float().to(opt.device)
            # print(item, label)

            # model predict
            prediction_score, hidden_state = model(item, hidden=hidden_state)
            # print(prediction_score.item(), label.item())

            # calculate each record loss
            loss += criterion(prediction_score.view_as(label), label)

            # form predict vector to calc auc
            gt_label.append(label.item())
            pred_label.append(prediction_score.item())

        # if np.sum(gt_label) == len(gt_label) or np.sum(gt_label) == 0:
        #     continue
        all_pred.append(pred_label)
        all_target.append(gt_label)

        if len(gt_label) != 0:
            loss /= len(gt_label)
            # auc = roc_auc_score(gt_label, pred_label)
            auc_meter.add(auc)
            loss_meter.add(loss.item())

        val_loss_list.append(str(loss_meter.value()[0])) # 训练到目前为止所有的loss平均值

        if opt.vis:
            vis.plot("valid_loss", loss_meter.value()[0])
            vis.plot("valid_auc", auc_meter.value()[0])

    if opt.vis:
        vis.log("epoch:{epoch}, valid_loss:{loss:.5f}, valid_auc:{auc:.5f}".format(epoch=epoch,
                                                                            loss=loss_meter.value()[0],
                                                                            auc=auc_meter.value()[0]))

    pred_1d = np.concatenate(all_pred, axis=0)
    target_1d = np.concatenate(all_target, axis=0)
    
    auc = roc_auc_score(target_1d, pred_1d)
    # print('test overall auc = ', auc)

    # with open('./test_is_text=False', mode='a') as file:
    #     for i in range(len(target_1d)):
    #         file.write(str(target_1d[i]))
    #         file.write(', ')
    #     file.write('\n')
    #     for i in range(len(pred_1d)):
    #         file.write(str(np.round(pred_1d[i], decimals=8)))
    #         file.write(', ')
    #     file.write('\n')
    #     file.write(str(auc))
    #     file.write('\n')

    return loss_meter, auc, val_loss_list