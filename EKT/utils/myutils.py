import torch.nn.utils.rnn as rnn_utils
import torch
import numpy as np
import time
from sklearn.metrics import roc_auc_score

def parse_csvdata(csv_path):
    practice_number_list = []
    knowledge_matrix = []
    result_matrix = []

    with open(csv_path) as f:
        for idx, line in enumerate(f.readlines()):
            if idx % 3 == 0:
                per_practice_number = int(line)
                practice_number_list.append(per_practice_number)
            if (idx - 1) % 3 == 0:
                per_knowledge_list = list(map(lambda x: int(x), line[:-1].split(',')))
                knowledge_matrix.append(per_knowledge_list)
            if (idx - 2) % 3 == 0:
                per_result_list = list(map(lambda x: int(x), line[:-1].split(',')))
                result_matrix.append(per_result_list)

    return practice_number_list, knowledge_matrix, result_matrix


def knowledge2vec(q_number, length, raw_seq, label):
    # if len(seq) != len(label):
    #     print("The length of data and label is not match!")
    seq = []
    vec_label = []

    for idx in range(len(raw_seq) - 1):
        k = int(raw_seq[idx])

        if k == 0 or idx >= length - 1:
            seq.append(0)
            vec_label.append([0, 0])
        else:
            l = int(label[idx])
            if l == 0:
                tmp_seq = k

            else:
                tmp_seq = q_number + k
            nk = int(raw_seq[idx+1])
            nl = int(label[idx+1])
            tmp_label_vec = [nk, nl]

            seq.append(tmp_seq)
            vec_label.append(tmp_label_vec)

    return length - 1, np.array(seq), np.array(vec_label)


def collate_fn(data):
    data.sort(key = lambda x: x[0], reverse=True)
    q_numbers = [x[0] for x in data]
    lengths = [x[1] for x in data]
    seqs = [x[2] for x in data]
    labels = [x[3] for x in data]

    seqs = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=0)

    data_lengths = []
    vec_seqs = [] # [batch, sequence_len, input_size]
    vec_labels = []
    for q_number, length, raw_seq, label in zip(q_numbers, lengths, seqs, labels):
        length, seq, vec_label = knowledge2vec(q_number, length, raw_seq, label)
        data_lengths.append(length)
        vec_seqs.append(seq)
        vec_labels.append(vec_label)

    data_lengths = torch.from_numpy(np.array(data_lengths, dtype=np.longlong))
    vec_seqs = torch.from_numpy(np.array(vec_seqs, dtype=np.longlong))
    vec_labels = torch.from_numpy(np.array(vec_labels, dtype=np.longlong))

    return data_lengths, vec_seqs, vec_labels


def collate_fn_max200(data):
    data.sort(key = lambda x: x[0], reverse=True)
    lengths = [x[0] for x in data]
    seqs = [x[1] for x in data]
    labels = [x[2] for x in data]

    seqs.append(torch.zeros(201))
    labels.append(torch.zeros(201))

    seqs = rnn_utils.pad_sequence(seqs, batch_first=True, padding_value=0)
    seqs = seqs[:-1, :]

    data_lengths = []
    vec_seqs = [] # [batch, sequence_len, input_size]
    vec_labels = []
    for length, raw_seq, label in zip(lengths, seqs, labels):
        length, seq, vec_label = knowledge2vec(length, raw_seq, label)
        data_lengths.append(length)
        vec_seqs.append(seq)
        vec_labels.append(vec_label)

    data_lengths = torch.from_numpy(np.array(data_lengths, dtype=np.longlong))
    vec_seqs = torch.from_numpy(np.array(vec_seqs, dtype=np.longlong))
    vec_labels = torch.from_numpy(np.array(vec_labels, dtype=np.longlong))

    return data_lengths, vec_seqs, vec_labels


def adjust_lr(opt, optimizer, epoch, current_loss):
    lr = opt.lr
    # if ((epoch + 1) % opt.decay_every_epoch) == 0:
    if (current_loss > opt.previous_loss) or ((epoch + 1) % opt.decay_every_epoch) == 0:
        lr = opt.lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    opt.lr = lr
    opt.previous_loss = current_loss

    '''
    lr = opt.lr * (opt.lr_decay ** ((epoch+1) // opt.decay_every_epoch))
    # if (loss_meter.value()[0] > previous_loss) or ((epoch + 1) % 10) == 0:
    if ((epoch + 1) % opt.decay_every_epoch) == 0:
        lr = lr * opt.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # previous_loss = train_loss_meter.value()[0]
    '''
    return lr


def save_model_weight(opt, model, optimizer, epoch, lr, is_final=False, is_CV=False):
    if is_CV:
        if is_final:
            prefix = opt.save_prefix + "DKT_final_params{}_{}_".format(str(opt.num_layers), str(opt.weight_decay))
        else:
            prefix = opt.save_prefix + "DKT_epoch{}_params{}_{}_".format(epoch, str(opt.num_layers),
                                                                         str(opt.weight_decay))
    else:
        if is_final:
            prefix = opt.save_prefix + "DKT_final_"
        else:
            prefix = opt.save_prefix + "DKT_epoch{}_".format(epoch)

    if opt.issave:
        file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict(),
            "lr": lr
        }
        torch.save(checkpoint, file_name)


def save_loss_file(opt, epoch, train_loss_list, val_loss_list, test_loss_list):
    if opt.issave:
        with open(opt.train_loss_path, 'a') as f:
            f.write("\nepoch_{}\n".format(epoch))
            f.write('\n'.join(train_loss_list))
        with open(opt.val_loss_path, 'a') as f:
            f.write("\nepoch_{}\n".format(epoch))
            f.write('\n'.join(val_loss_list))
        with open(opt.test_loss_path, 'a') as f:
            f.write("\nepoch_{}\n".format(epoch))
            f.write('\n'.join(test_loss_list))


def Variable(*args, **kwargs):
    use_cuda = False
    v = torch.autograd.Variable(*args, **kwargs)
    if use_cuda:
        v = v.cuda()
    return v

def load_embedding(filename):
    f = open(filename, encoding='utf-8')
    wcnt, emb_size = next(f).strip().split(' ')
    wcnt = int(wcnt)
    emb_size = int(emb_size)

    words = []
    embs = []
    for line in f:
        fields = line.strip().split(' ')
        word = fields[0]
        emb = np.array([float(x) for x in fields[1:]])
        words.append(word)
        embs.append(emb)

    embs = np.asarray(embs)
    return wcnt, emb_size, words, embs
