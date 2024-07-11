import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import utils as U

import wandb

import os


class LSTM_pred(torch.nn.Module):
    def __init__(self, input_dim = 512, hid_dim=256, n_layers=4, teacher_forcing = 1.0):
        super().__init__()
        
        self.hid_dim = hid_dim

        self.lstm = torch.nn.LSTM(input_dim, hid_dim, num_layers=n_layers, bidirectional=False, batch_first=True)#dropout = 0.2

        self.actv = torch.nn.ELU()

        self.teacher_forcing = teacher_forcing

    
    def forward(self, x, time_step=1):
        seq_len = x.shape[1]

        y, _ = self.lstm(x[:, 0, :][:, None, :])
        y = self.actv(y)

        for i in range(1, seq_len):
            if np.random.random() <= self.teacher_forcing:
                w, _ = self.lstm(x[:, i, :][:, None, :])
                w = self.actv(w)
                y = torch.cat((y, w), dim=1)
            else:
                w, _ = self.lstm(y[:, i-1, :][:, None, :])
                w = self.actv(w)
                y = torch.cat((y, w), dim=1)

        for _ in range(time_step - 1):
            w, _ = self.lstm(y[:, -1, :][:, None, :])
            w = self.actv(w)
            y = torch.cat((y, w), dim=1)
        
        return y, None, None
    
def train(model, autoencoder, train_loader, validation_loader, criterion, optimizer, scheduler,  device, c):#FIXME removed config
    for epoch in range(c['epochs']):
        loss_epoch, loss_img_epoch, loss_feat_epoch = train_lstm(model, autoencoder, train_loader, criterion, optimizer, epoch, device, c['img_loss_weights'], c['time_step'])
        val_loss, rmse_dict = evaluate_lstm(model, autoencoder, validation_loader, criterion, device, c['img_loss_weights'], c['time_step'])

        if criterion[4]:
            wandb.log({"Total Train Loss": loss_epoch, "Img Train Loss": loss_img_epoch, "Feature Train Loss": loss_feat_epoch, "Validation Loss": val_loss}|rmse_dict)
        else:
            wandb.log({"Train Loss": loss_epoch, "Validation Loss": val_loss}|rmse_dict)
    
        if scheduler:
            scheduler.step()
        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), os.path.join(c['dir_path'], f"model_e{epoch}.pt"))
    return

def train_lstm(model, autoencoder, train_loader, criterion, optimizer, epoch, device, l_w, time_step):
    model.train()
    #freeze the autoencoder
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
    # Training loop
    losses = []
    losses_feat = []
    losses_img =  []
    progress_bar = tqdm(total=len(train_loader), unit='step')
    for i, images in enumerate(train_loader):
        images = images["depth8"].to(device)

        optimizer.zero_grad()

        
        batch_len = images.shape[0]
        seq_len = images.shape[1]
        images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
        latent = autoencoder.encode(images)
        new_batch = latent.reshape(batch_len,seq_len, latent.shape[1]*latent.shape[2]*latent.shape[3])

        images = images.reshape(batch_len, seq_len, images.shape[1], images.shape[2], images.shape[3])

        output, hid, _ = model(new_batch[:,:-time_step, :], time_step)##TODO


        #loss = criterion(output, new_batch[:, 1:, :])

        #if with_decoder == True:
        
        output_img = output.reshape(output.shape[0]*output.shape[1], 512, 1, 1)
        output_img = autoencoder.decode(output_img)
        #output = output.reshape(batch_len, seq_len - 1, output.shape[1], output.shape[2], output.shape[3])

        #images = images.reshape(batch_len, seq_len, images.shape[1], images.shape[2], images.shape[3])
        gt_images = images[:, 1:, :]
        gt_images = gt_images.reshape(batch_len*(seq_len-1), gt_images.shape[2], gt_images.shape[3], gt_images.shape[4])
        loss = l_w[0]*criterion[0](output_img, gt_images) + l_w[1]*criterion[1](output_img, gt_images) + l_w[2]*criterion[2](output_img, gt_images)

        """if criterion[3]:
            gt_images2 = images[:, :-1, :]
            gt_images2 = gt_images2.reshape(batch_len*(seq_len-1), gt_images2.shape[2], gt_images2.shape[3], gt_images2.shape[4])

            weight_loss_img_diff = criterion[3][0](gt_images.detach(), gt_images2)#for mssim
            if criterion[3][2]:
                weight_loss_img_diff = torch.mean(weight_loss_img_diff, dim=(1, 2, 3))
                weight_loss_img_diff = weight_loss_img_diff/torch.max(weight_loss_img_diff)
            
            loss_img_diff = l_w[0]*criterion[0](output_img, gt_images2) + l_w[1]*criterion[1](output_img, gt_images2) + l_w[2]*criterion[2](output_img, gt_images2)
            loss = loss + criterion[3][1]*weight_loss_img_diff*loss_img_diff"""

        #loss = torch.mean(loss)
        losses_img.append(torch.mean(loss).item())

        if criterion[4]:
            loss_feat = criterion[4][0](output, new_batch[:, 1:, :])
            losses_feat.append(torch.mean(loss_feat).item())
            #loss_feat = torch.mean(loss_feat)
            loss_feat = torch.mean(loss_feat, dim=(2))
            loss_feat = loss_feat.reshape(batch_len*(seq_len-1))
            loss = loss + criterion[4][1]*loss_feat

        if criterion[3]:
            gt_images2 = images[:, :-1, :]
            gt_images2 = gt_images2.reshape(batch_len*(seq_len-1), gt_images2.shape[2], gt_images2.shape[3], gt_images2.shape[4])

            weight_loss_img_diff = criterion[3][0](gt_images.detach(), gt_images2)#for mssim
            if criterion[3][2]:
                weight_loss_img_diff = torch.mean(weight_loss_img_diff, dim=(1, 2, 3))
                weight_loss_img_diff = weight_loss_img_diff/torch.max(weight_loss_img_diff)
            
            loss_img_diff = l_w[0]*criterion[0](output_img, gt_images2) + l_w[1]*criterion[1](output_img, gt_images2) + l_w[2]*criterion[2](output_img, gt_images2)
            loss = loss - criterion[3][1]*weight_loss_img_diff*loss_img_diff

            if criterion[4]:
                loss_feat_diff = criterion[4][0](output, new_batch[:, :-1, :])
                loss_feat_diff = torch.mean(loss_feat_diff, dim=(2))
                loss_feat_diff = loss_feat_diff.reshape(batch_len*(seq_len-1))
                #It is weighted as the image diff loss with criterion[3][1]
                loss = loss - criterion[4][1]*criterion[3][1]*weight_loss_img_diff*loss_feat_diff
        
        loss = torch.mean(loss)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        running_loss = np.mean(losses)#TOCHECK L1 norm?

        #Progress Bar
        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar.set_postfix(loss=running_loss)
        progress_bar.update(1)
    
    # endfor batch
    l_f = None
    if criterion[4]:
        l_f = np.mean(losses_feat)
    
    return np.mean(losses), np.mean(losses_img), l_f


def evaluate_lstm(model, autoencoder, validation_loader, criterion, device, l_w, time_step):
    
    model.eval()
    autoencoder.eval()
    # Training loop

    num_batches = len(validation_loader)
    losses = []
    rmse = []
    rmse_back = []
    progress_bar = tqdm(total=len(validation_loader), unit='step')
    with torch.no_grad():
        for i, images in enumerate(validation_loader):
            images = images["depth8"].to(device)

            batch_len = images.shape[0]
            seq_len = images.shape[1]
            images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
            latent = autoencoder.encode(images)
            images = images.reshape(batch_len, seq_len, images.shape[1], images.shape[2], images.shape[3])
            new_batch = latent.reshape(batch_len,seq_len, latent.shape[1]*latent.shape[2]*latent.shape[3])

            output, hid, _ = model(new_batch[:,:-time_step, :], time_step)#one step prediction

            output_img = output.reshape(output.shape[0]*output.shape[1], 512, 1, 1)
            output_img = autoencoder.decode(output_img)

            gt_images = images[:, 1:, :]
            gt_images = gt_images.reshape(batch_len*(seq_len-1), gt_images.shape[2], gt_images.shape[3], gt_images.shape[4])
            #loss = criterion(output, new_batch[:, 1:, :])
            loss = l_w[0]*criterion[0](output_img, gt_images) + l_w[1]*criterion[1](output_img, gt_images) + l_w[2]*criterion[2](output_img, gt_images)

            if criterion[4]:
                loss_feat = criterion[4][0](output, new_batch[:, 1:, :])
                loss_feat = torch.mean(loss_feat, dim=(2))
                loss_feat = loss_feat.reshape(batch_len*(seq_len-1))
                loss = loss + criterion[4][1]*loss_feat

            loss = torch.mean(loss)
            losses.append(loss.item())
            running_loss = np.mean(losses)

            gt_images = gt_images.reshape(batch_len, (seq_len-1), gt_images.shape[1], gt_images.shape[2], gt_images.shape[3])
            output_img = output_img.reshape(batch_len, (seq_len-1), output_img.shape[1], output_img.shape[2], output_img.shape[3])

            rmse.append(torch.sqrt(torch.mean((output_img - gt_images)**2, dim=(0,2,3,4)))[None, :])
            rmse_back.append(torch.sqrt(torch.mean((output_img - images[:, :-1, :])**2, dim=(0,2,3,4)))[None, :])

            """for i in range(time_step):
                rmse[i].append(torch.sqrt(torch.mean((output_img[:-time_step+i] - images[:, :time_step+i, :])**2, dim=(3,4))).squeeze())"""
            """
            reconvert features in images 
            output = output.reshape(output.shape[0]*output.shape[1], 1024, 2, 2)
            output = autoencoder.decode(output)
            output = output.reshape(batch_len, seq_len-1, output.shape[1], output.shape[2], output.shape[3])"""

            #Progress Bar
            progress_bar.set_description(f"Validation")
            progress_bar.set_postfix(loss=running_loss)
            progress_bar.update(1)


    # endfor batch
    rmse = torch.cat(rmse, dim=0)
    rmse = torch.mean(rmse, dim=0).cpu().numpy()
    rmse_dict = {"RMSE_tot": np.mean(rmse)}

    rmse_back = torch.cat(rmse_back, dim=0)
    rmse_back = torch.mean(rmse_back, dim=0).cpu().numpy()
    rmse_dict["RMSE_back_tot"] = np.mean(rmse_back)

    for i in range(time_step):
        rmse_dict[f"RMSE_step_{i+1}"] = rmse[-time_step+i]
        rmse_dict[f"RMSE_back_step_{i+1}"] = rmse_back[-time_step+i]

        
    
    return np.mean(losses), rmse_dict




