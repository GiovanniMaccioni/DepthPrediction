import torch
from tqdm import tqdm

import utils as U

import wandb

import torchvision

import os

def train(model, train_loader, validation_loader, criterion, optimizer, scheduler, device, c):#FIXME removed config
    for epoch in range(c["epochs"]):
        loss_epoch, image_logging_t = train_batch_depth_estimation(model, train_loader, criterion, optimizer, epoch, device, c["loss_weigths"])
        val_loss, image_logging_v, rmse = evaluate_batch(model, validation_loader, criterion, device, c["loss_weigths"])
        if scheduler != None:
            scheduler.step()

        wandb.log({"Train Loss": loss_epoch, "Validation Loss": val_loss, "RMSE":rmse}|image_logging_t|image_logging_v)

        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), os.path.join(c["dir_path"], f"model_e{epoch}.pt"))


    return 

def train_batch_depth_estimation(model, train_loader, criterion, optimizer, epoch, device, l_w):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    #Progress Bar
    progress_bar = tqdm(total = len(train_loader), unit='step')
    for i, images in enumerate(train_loader):
        images = images["depth8"].to(device)
        optimizer.zero_grad()
        out, _ = model(images)

        """#TODO create a mask for zero depth values
        out = torch.where(images[:,0,:][:,None,:] > 0, out, 0.0)"""

        #loss = criterion[0](out, images) + criterion[1](out, images) + criterion[2](out, images)
        """L = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])[None,None, :].to(device)
        edges = torch.nn.functional.conv2d(images, L, padding=1)
        edges = (edges - edges.min())/(edges.max()-edges.min())

        images = images+edges
        
        images = (images - images.min())/(images.max()-images.min())"""

        #loss = criterion(out, images)
        loss = l_w[0]*criterion[0](out, images) + l_w[1]*criterion[1](out, images) + l_w[2]*criterion[2](out, images)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK L1 norm?

        #Progress Bar
        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar.set_postfix(loss=running_loss)
        progress_bar.update(1)

        if i == 0:
            max_ = images[0, 0, :].detach().cpu().max().float()
            min_ = images[0, 0, :].detach().cpu().min().float()
            image = torchvision.transforms.functional.to_pil_image(images[0, 0, :], mode=None)
            gt = wandb.Image(image, caption=f"Ground truth, range {min_:0.3f}, {max_:0.3f}")

            max_ = out[0].detach().cpu().max().float()
            min_ = out[0].detach().cpu().min().float()
            image = torchvision.transforms.functional.to_pil_image(out[0], mode=None)
            predicted = wandb.Image(image, caption=f"Predicted, range {min_:0.3f}, {max_:0.3f}")
            
            imgs = [gt, predicted]

            image_logging = {"Train": imgs}

    return running_loss, image_logging


def evaluate_batch(model, loader, criterion, device, l_w):
    model.eval()
    running_loss = 0
    running_RMSE = 0


    num_batches = len(loader)
    #len_data = len(loader)*(loader.batch_size)

    #Progress Bar
    progress_bar = tqdm(total = len(loader), unit='step')

    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images["depth8"].to(device)
            #images = images.reshape((images.shape[0], images.shape[1]*images.shape[2], images.shape[3], images.shape[4] ))#----> NOT NEEDED FOR 3D CONVOLUTIONS
            out, _ = model(images)
            #Note that the normalization is already applied to the groundtruth as well as the 
            # sequence in input!!
            #FIXME this acces to the image is bad thought


            """#TODO create a mask for zero depth values
            out = torch.where(images[:,0,:][:,None,:] > 0, out, 0.0)"""
            loss = l_w[0]*criterion[0](out, images) + l_w[1]*criterion[1](out, images) + l_w[2]*criterion[2](out, images)
            
            running_loss += loss.item()/num_batches

            running_RMSE += torch.sqrt(torch.mean((out - images)**2))

            #Progress Bar
            progress_bar.set_description(f"Validation")
            progress_bar.set_postfix(loss=running_loss)
            progress_bar.update(1)

            if i == 0:
                max_ = images[0, 0, :].detach().cpu().max().float()
                min_ = images[0, 0, :].detach().cpu().min().float()
                image_gt = torchvision.transforms.functional.to_pil_image(images[0, 0, :], mode=None)
                gt = wandb.Image(image_gt, caption=f"Ground truth, range {min_:0.3f}, {max_:0.3f}")

                max_ = out[0].detach().cpu().max().float()
                min_ = out[0].detach().cpu().min().float()
                image_out = torchvision.transforms.functional.to_pil_image(out[0], mode=None)
                predicted = wandb.Image(image_out, caption=f"Predicted, range {min_:0.3f}, {max_:0.3f}")

                #image = torchvision.transforms.functional.to_pil_image(torch.mean(latent[0], dim=0), mode=None)
                #mean_f = wandb.Image(image, caption=f"Channel Mean Feature Map After encoder")
                
                imgs = [gt, predicted]#, mean_f]

                image_logging = {"Validation": imgs}

    return running_loss, image_logging, running_RMSE/num_batches