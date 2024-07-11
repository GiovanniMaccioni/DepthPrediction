import torch
import model as M
import data as D
import autoencoder_train as T
import lstm_train as L
import utils as U
import os

from losses import *

def run(config):
    if not os.path.exists(config['dir_path']):
            os.makedirs(config['dir_path'])


    U.set_reproducibility(config['seed'])
    #Hyperparameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #Load Autoencoder
    encoder = M.Encoder()
    decoder = M.Decoder()
    autoencoder = M.Autoencoder_conv(encoder, decoder).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join("./runs_aut_fin/run19b", "model.pt")))

    trainset =  D.RecDataset("./data/dataset", "train", img_size=config["train_img_size"], sequence_length=config['sequence_length']+config['time_step'], img_transform=["img_transform"], seq_transform = config["seq_transform"], fps_augmentation=config["fps_augmentation"])
    trainset.train()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'],
                                            shuffle=True, num_workers=12,
                                            worker_init_fn=D.init_worker, drop_last=True)

    valset =  D.RecDataset("./data/dataset", "train", img_size=config["val_img_size"], sequence_length=config['sequence_length']+config['time_step'])
    valset.eval()
    valloader =  torch.utils.data.DataLoader(valset, batch_size=16,
                                        shuffle=False, num_workers=12,
                                        worker_init_fn=D.init_worker, drop_last=True)

    with T.wandb.init(project=f"lstm_avoid_copying2", name=f"seed{config['seed']}_{os.path.basename(config['dir_path'])}", config = config):#, mode="disabled"

        
        lstm = L.LSTM_pred(input_dim = 512, hid_dim=config['hid_dim'], n_layers = config['n_layers'], teacher_forcing=config["teacher_forcing"]).to(device)
        optimizer_lstm = torch.optim.Adam(lstm.parameters(), config['lr'])

        if config['scheduler'] == "StepLR":
            scheduler_lstm = torch.optim.lr_scheduler.StepLR(optimizer_lstm, step_size=config["scheduler_parameters"][0], gamma=config["scheduler_parameters"][1])
        else:
            scheduler_lstm = None

        criterion = [MSSIMLoss(n_channels=1, window_size=11, reduction='none').to(device), BerHULoss(reduction='none'), SOBELLoss(device, reduction='none')]

        if config['img_diff_loss']=="L1":
            criterion.append([torch.nn.L1Loss(reduction='none'), config['img_diff_loss_weight'], True])
        elif config['img_diff_loss']=="MSE":
            criterion.append([torch.nn.MSELoss(reduction='none'), config['img_diff_loss_weight'], True])
        elif config['img_diff_loss']=="MSSIM":
            criterion.append([MSSIMLoss(n_channels=1, window_size=11, reduction='none').to(device), config['img_diff_loss_weight'], False])
        else:
            criterion.append(None)


        if config['feature_loss']=="L1":
            criterion.append([torch.nn.L1Loss(reduction='none'), config['feature_loss_weight']])
        elif config['feature_loss']=="MSE":
            criterion.append([torch.nn.MSELoss(reduction='none'), config['feature_loss_weight']])
        else:
            criterion.append(None)


        T.wandb.watch(lstm, criterion, log="all", log_freq=1)
        L.train(lstm, autoencoder, trainloader, valloader, criterion, optimizer_lstm, scheduler_lstm, device, config)


        valset =  D.RecDataset("./data/dataset", "train", img_size=config["val_img_size"], sequence_length=config['sequence_length']+config['time_step'])
        valset.eval()
        valloader =  torch.utils.data.DataLoader(valset, batch_size=1,
                                            shuffle=False, num_workers=12,
                                            worker_init_fn=D.init_worker, drop_last=True)
        
        U.save_run_results_lstm(lstm, autoencoder, config['dir_path'], valloader, device, config)

    T.wandb.finish()
    return


