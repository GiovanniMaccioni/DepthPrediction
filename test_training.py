import torch
import models as M
import data as D
import train as T
import utils as U

import torchvision

config ={
    'batch_size': 128,
    'seed': 10,
    'lr': 3e-5,
    'sequence_length':1,
    'img_size': (512, 432), #(512, 424),None
    'img_norm': "min_max",#"mean_std" "min_max"
    'epochs': 31
}

from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

U.set_reproducibility(config['seed'])

#Hyperparameters
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trainset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=config["img_size"], sequence_length=1, norm_type=config["img_norm"])
trainset.train()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config["batch_size"],
                                        shuffle=True, num_workers=16,
                                        worker_init_fn=D.init_worker, drop_last=True)


valset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=config["img_size"], sequence_length=1, norm_type=config["img_norm"])
valset.eval()
valloader =  torch.utils.data.DataLoader(valset, batch_size=config["batch_size"],
                                    shuffle=False, num_workers=16,
                                    worker_init_fn=D.init_worker, drop_last=True)



#TOCHECK I don't know if I have to do the XYZ transformation as said in the paper
with T.wandb.init(project=f"experiment25-Resnet_encoder", name=f"seed{config['seed']}_resnet50_encoder_dec_pretrained_encfreeze", config = config):#, mode="disabled"

    #encoder = M.Encoder()
    encoder = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.fc = torch.nn.Linear(encoder.fc.in_features, 1000)
    for param in encoder.fc.parameters():
        param.requires_grad = True
    decoder = M.Decoder()
    model = M.Autoencoder_conv(encoder, decoder).to(device)

    #I will test depth estimation first. To change the test for now, change the function call in the train function in train.py

    #criterion = torch.nn.L1Loss()
    #criterion = SSIMLoss(n_channels=1).cuda() #if you need GPU support
    criterion = torch.nn.HuberLoss(delta = 0.2)
    #criterion = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), config['lr'])

    T.wandb.watch(model, criterion, log="all", log_freq=1)

    T.train(model, trainloader, valloader, criterion, optimizer, device, config['epochs'])

    for param in encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), 3e-5)

    T.train(model, trainloader, valloader, criterion, optimizer, device, 20)
    

T.wandb.finish()



