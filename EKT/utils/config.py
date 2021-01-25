import torch

class Config:
    isdev = False

    vis = False if isdev else True
    env = "DKT"
    plot_every_iter = 20

    gpu = False if isdev else True
    device = torch.device('cuda') if gpu == True and torch.cuda.is_available() else torch.device('cpu')
    # print(device)

    data_source = 'assist2009_updated'
    fold_dataset = False
    cv_times = 1
    model_name = "EKT"
    ablation = "None"

    # batch_size = 100
    batch_size = 4 if isdev else 64
    num_workers = 1 if isdev else 8

    H = 15
    embed_dim = 225
    hidden_dim = 225
    output_dim = 110
    k_frames = 8
    input_dim = 2 * 110

    num_layers = 1
    lr = 0.001
    lr_decay = 0.3
    decay_every_epoch = 5
    previous_loss = 1e10
    weight_decay = 0.00005 # 1e-5有效(lr = 1e-3)，但如果不给大一点，容易过拟合
    max_epoch = 50

    issave = False
    save_prefix = "checkpoints/"
    save_every = 5
    train_loss_path = save_prefix+"train_loss_list.txt"
    val_loss_path = save_prefix+"val_loss_list.txt"
    test_loss_path = save_prefix+"test_loss_list.txt"

    model_path = None

    # EKT option
    is_text = True
