import torch
import model as M
import data as D
import autoencoder_train as T
import utils as U
from losses import *
import os

import torchvision.transforms.functional as TVF

"""import torchvision"""

from torchvision.transforms import v2

def run(config):

    if not os.path.exists(config['dir_path']):
        os.makedirs(config['dir_path'])

    #print(os.path.basename(config['dir_path']))
    with open(os.path.join("./", "config.txt"), 'w') as f:
        print(config, file=f)

    U.set_reproducibility(config['seed'])

    #Hyperparameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_transforms = None
    if config['img_transform']:
        if config['random_crop']:
            train_transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomPerspective(distortion_scale=0.1, p=0.5),#prima 0.1
                v2.RandomRotation(degrees=(-45, 45)),
                v2.RandomCrop(256)
            ])
        else:
            train_transforms = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomPerspective(distortion_scale=0.1, p=0.5, interpolation=TVF.InterpolationMode.NEAREST),
                v2.RandomRotation(degrees=(-45, 45))
            ])

    trainset =  D.RecDataset("./data/dataset", "train", img_size=config["train_img_size"], sequence_length=1, img_transform = train_transforms)
    trainset.train()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                            shuffle=True, num_workers=10,
                                            worker_init_fn=D.init_worker, drop_last=True)


    valset =  D.RecDataset("./data/dataset", "train", img_size=config["val_img_size"], sequence_length=1)
    valset.eval()
    valloader =  torch.utils.data.DataLoader(valset, batch_size=config["batch_size"],
                                        shuffle=False, num_workers=10,
                                        worker_init_fn=D.init_worker, drop_last=True)



    #TOCHECK I don't know if I have to do the XYZ transformation as said in the paper
    with T.wandb.init(project=f"autoencoder_hyper_fin", name=f"seed{config['seed']}_{os.path.basename(config['dir_path'])}", config = config):#, mode="disabled"

        encoder = M.Encoder()
        decoder = M.Decoder()
        model = M.Autoencoder_conv(encoder, decoder).to(device)

        print("Encoder #parameters: ", sum(p.numel() for p in encoder.parameters()))
        print("Decoder #parameters: ", sum(p.numel() for p in decoder.parameters()))

        #I will test depth estimation first. To change the test for now, change the function call in the train function in train.py

        criterion = [MSSIMLoss(n_channels=1, window_size=11).to(device), BerHULoss(), SOBELLoss(device)]


        optimizer = torch.optim.Adam(model.parameters(), config['lr'])
        if config['scheduler'] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["scheduler_parameters"][0], gamma=config["scheduler_parameters"][1])
        else:
            scheduler = None

        T.wandb.watch(model, criterion, log="all", log_freq=1)

        T.train(model, trainloader, valloader, criterion, optimizer, scheduler, device, config)

        #Save sequences
        valset =  D.RecDataset("./data/dataset", "train", img_size=config["val_img_size"], sequence_length=10)
        valset.eval()
        valloader =  torch.utils.data.DataLoader(valset, batch_size=1,
                                            shuffle=False, num_workers=1,
                                            worker_init_fn=D.init_worker, drop_last=True)

        U.save_run_results(model, config['dir_path'], valloader, device, config)
        
    T.wandb.finish()

    return






