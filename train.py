import torch
from tqdm import tqdm

import utils as U

import wandb

import torchvision

def train(model, train_loader, validation_loader, criterion, optimizer, device, epochs):#FIXME removed config
    for epoch in range(epochs):
        loss_epoch, image_logging_t = train_batch_depth_estimation(model, train_loader, criterion, optimizer, epoch, device)
        val_loss, image_logging_v = evaluate_batch(model, validation_loader, criterion, device)

        wandb.log({"Train Loss": loss_epoch, "Validation Loss": val_loss}|image_logging_t|image_logging_v)

    return 

def train_batch_depth_estimation(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    #Progress Bar
    progress_bar = tqdm(total = len(train_loader), unit='step')
    for i, images in enumerate(train_loader):
        images = images["depth8"].to(device)
        #images = images.squeeze()#FIXME added to unify channels and sequence length; to sub it for reshape
        #images.shape: [batch_size, sequence_length, channels, height, width]
        images = images.reshape((images.shape[0], images.shape[1]*images.shape[2], images.shape[3], images.shape[4] ))#----> NOT NEEDED FOR 3D CONVOLUTIONS
        optimizer.zero_grad()
        out, latent = model(images)
        loss = criterion(out, images[:, 0, :][:, None])#Selected the last frame of the sequence to reconstruct it. 0 to select the first frame(depth reconstruction)
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
            gt = wandb.Image(image, caption=f"Ground truth, range {min_:0.3f}-{max_:0.3f}")

            max_ = out[0].detach().cpu().max().float()
            min_ = out[0].detach().cpu().min().float()
            image = torchvision.transforms.functional.to_pil_image(images[0, 0, :], mode=None)
            predicted = wandb.Image(image, caption=f"Predicted, range {min_:0.3f}-{max_:0.3f}")
            image_logging = {"Ground Truth": gt, "Predicted": predicted}

    return running_loss, image_logging

def train_batch_depth_prediction(model, train_loader, criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)#TOCHECK Maybe it returns the len - sequence length because of the bad practice in the dataset class

    for images in tqdm(train_loader, desc=f'Training epoch {epoch}', leave=True):
        images = images.to(device)
        optimizer.zero_grad()
        out, latent = model(images)
        loss = criterion(out, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK L1 norm?
        
    
    return running_loss

def evaluate_batch(model, loader, criterion, device):
    model.eval()
    accuracy = 0
    running_loss = 0
    num_batches = len(loader)
    #len_data = len(loader)*(loader.batch_size)

    #Progress Bar
    progress_bar = tqdm(total = len(loader), unit='step')

    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images["depth8"].to(device)
            images = images.reshape((images.shape[0], images.shape[1]*images.shape[2], images.shape[3], images.shape[4] ))#----> NOT NEEDED FOR 3D CONVOLUTIONS
            out, latent = model(images)
            #Note that the normalization is already applied to the groundtruth as well as the 
            # sequence in input!!
            #FIXME this acces to the image is bad thought
            loss = criterion(out, images[:, 0, :][:, None])

            #U.show_depths([images[0], out[0]])

            running_loss += loss.item()/num_batches

            #Progress Bar
            progress_bar.set_description(f"Validation")
            progress_bar.set_postfix(loss=running_loss)
            progress_bar.update(1)

            if i == 0:
                max_ = images[0, 0, :].detach().cpu().max().float()
                min_ = images[0, 0, :].detach().cpu().min().float()
                image_gt = torchvision.transforms.functional.to_pil_image(images[0, 0, :], mode=None)
                gt = wandb.Image(image_gt, caption=f"Ground truth, range {min_:0.3f}-{max_:0.3f}")

                max_ = out[0].detach().cpu().max().float()
                min_ = out[0].detach().cpu().min().float()
                image_out = torchvision.transforms.functional.to_pil_image(out[0], mode=None)
                predicted = wandb.Image(image_out, caption=f"Predicted, range {min_:0.3f}-{max_:0.3f}")
                image_logging = {"Ground Truth": gt, "Predicted": predicted}

    return running_loss, image_logging