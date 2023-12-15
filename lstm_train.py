import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import data as D

import mae.mae.models_mae as models_mae

class Autoencoder_MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(81920, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 128)

        self.linear5 = nn.Linear(128, 256)
        self.linear6 = nn.Linear(256, 1024)
        self.linear7 = nn.Linear(1024, 2048)
        self.linear8 = nn.Linear(2048, 81920)

        self.actv = nn.LeakyReLU(0.2)


    def forward(self, x):

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)
        x = self.linear3(x)
        x = self.actv(x)
        x = self.linear4(x)

        x = self.actv(x)
        x = self.linear5(x)
        x = self.actv(x)
        x = self.linear6(x)
        x = self.actv(x)
        x = self.linear7(x)
        x = self.actv(x)
        x = self.linear8(x)

        return x

class LSTM_pred(torch.nn.Module):
    def __init__(self, input_dim = 512, hid_dim=256, n_layers=4):
        super().__init__()
        
        self.hid_dim = hid_dim

        self.rnn = torch.nn.LSTM(input_dim, hid_dim, num_layers=n_layers, bidirectional=False, batch_first=True)
        """self.linear_in1 = torch.nn.Linear(81920, 1024)
        self.linear_in2 = torch.nn.Linear(1024, input_dim)"""
        self.linear_out1 = torch.nn.Linear(hid_dim, 1024)
        self.linear_out2 = torch.nn.Linear(1024, 81920)

        self.actv = torch.nn.ReLU()

    def forward(self, x):
        
        #input = [batch size, sequence_length, emb_dim]
        #hidden = [n directions*num_layers, batch size, hid dim]

        """x = self.linear_in1(x)
        x = self.actv(x)
        x = self.linear_in2(x)"""


        x, (hidden, cell) = self.rnn(x)

        x = self.linear_out1(x)
        x = self.actv(x)
        x = self.linear_out2(x)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #self.linear1 = nn.Linear(hid_dim, 512)
        
        #prediction = [batch size, output dim]
        
        return x, hidden, cell
    


def train_lstm(model, model_mae, criterion, optimizer,train_loader, epochs, device): #vocabulary, config,valloader, modeltitle= "_AV"

    model.train()
    # Training loop

    for epoch in range(epochs):

        losses = []
        progress_bar = tqdm(total=len(train_loader), unit='step')
        for i, images in enumerate(train_loader):
            images = images["depth8"].to(device)
            images = torch.cat((images, images, images), dim = 2)

            optimizer.zero_grad()

            new_batch = []
            cls_tokens = []
            with torch.no_grad():
                for seq in images:
                    latent, _, _= model_mae.forward_encoder(seq, mask_ratio = 0.75)
                    cls_tokens.append(latent[None, :, 0, :])
                    new_batch.append(latent[None,:, 1:, :])
            
            new_batch = torch.cat(new_batch)
            new_batch = new_batch.reshape((new_batch.shape[0], new_batch.shape[1], new_batch.shape[2]*new_batch.shape[3]))

            output, hid, _ = model(new_batch[:,:-1, :])

            loss = criterion(output[:, -1,:], new_batch[:, -1, :])
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            running_loss = np.mean(losses) #TOCHECK L1 norm?

            #Progress Bar
            progress_bar.set_description(f"Epoch {epoch}")
            progress_bar.set_postfix(loss=running_loss)
            progress_bar.update(1)
        
        # endfor batch
        print(f"Mean Loss: {np.mean(losses):10f}")

    return

def train_autoencoder(model, model_mae, criterion, optimizer,train_loader, epochs, device): #vocabulary, config,valloader, modeltitle= "_AV"

    model.train()
    # Training loop

    for epoch in range(epochs):

        losses = []
        progress_bar = tqdm(total=len(train_loader), unit='step')
        for i, images in enumerate(train_loader):
            images = images["depth8"].to(device)
            images = torch.cat((images, images, images), dim = 1)

            optimizer.zero_grad()

            new_batch = []
            cls_tokens = []
            with torch.no_grad():
                new_batch, _, _= model_mae.forward_encoder(images, mask_ratio = 0.75)
            
            new_batch = new_batch[:, 1:, :]
            new_batch = new_batch.reshape((new_batch.shape[0], new_batch.shape[1]*new_batch.shape[2]))

            output= model(new_batch)

            loss = criterion(output, new_batch)
            loss.backward()
            
            optimizer.step()

            losses.append(loss.item())

            running_loss = np.mean(losses) #TOCHECK L1 norm?

            #Progress Bar
            progress_bar.set_description(f"Epoch {epoch}")
            progress_bar.set_postfix(loss=running_loss)
            progress_bar.update(1)
        
        # endfor batch

    return

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


trainset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=(224,224), sequence_length=15, norm_type="min_max")
trainset.train()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                        shuffle=True, num_workers=16,
                                        worker_init_fn=D.init_worker, drop_last=True)

valset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=(224,224), sequence_length=15, norm_type="min_max")
valset.eval()
valloader =  torch.utils.data.DataLoader(valset, batch_size=2,
                                    shuffle=False, num_workers=16,
                                    worker_init_fn=D.init_worker, drop_last=True)


lstm = LSTM_pred(input_dim = 81920, hid_dim=1024, n_layers = 12).to(device)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), 3e-2)

"""aut = Autoencoder_MLP().to(device)
optimizer_aut = torch.optim.Adam(aut.parameters(), 3e-3)"""


criterion = torch.nn.L1Loss()


"""model_mae = models_mae.mae_vit_huge_patch14_dec512d8b()
temp = torch.load("./mae_huge_finetuned.pt", map_location='cpu')"""
model_mae = models_mae.mae_vit_base_patch16_dec512d8b_custom()
temp = torch.load("./mae_base_from_scratch.pt", map_location='cpu')
model_mae.load_state_dict(temp, strict=False)
model_mae = model_mae.to(device)

#train_lstm(lstm, model_mae, criterion, optimizer_lstm, trainloader, 10, device)
train_lstm(lstm, model_mae, criterion, optimizer_lstm, trainloader, 100, device)
