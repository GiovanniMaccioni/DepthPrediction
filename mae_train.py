import sys
import os
#import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

"""# check whether run in Colab
if 'google.colab' in sys.modules:
    print('Running in Colab.')
    !pip3 install timm==0.4.5  # 0.3.2 does not work in Colab
    !git clone https://github.com/facebookresearch/mae.git
    sys.path.append('./mae')
else:
"""
import mae.mae.models_mae as models_mae


import data as D
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# define the utils

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    #plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.imshow(torch.clip(image * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img).to(device)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x).to("cpu")

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()

def train(model, train_loader, optimizer, device, epochs):#FIXME removed config
    for epoch in range(epochs):
        loss_epoch = train_batch_depth_estimation(model, train_loader, optimizer, epoch, device)

    return model

def train_batch_depth_estimation(model, train_loader, optimizer, epoch, device):
    model.train()
    running_loss = 0
    num_batches = len(train_loader)
    #Progress Bar
    progress_bar = tqdm(total = len(train_loader), unit='step')
    for i, images in enumerate(train_loader):
        images = images["depth8"].to(device)
        images = torch.cat((images, images, images), dim = 1)
        #images = images.squeeze()#FIXME added to unify channels and sequence length; to sub it for reshape
        #images.shape: [batch_size, sequence_length, channels, height, width]
        #images = images.reshape((images.shape[0], images.shape[1]*images.shape[2], images.shape[3], images.shape[4] ))#----> NOT NEEDED FOR 3D CONVOLUTIONS
        optimizer.zero_grad()
        
        loss, y, mask = model(images, mask_ratio=0.75)

        #TODO create a mask for zero depth values

        loss.backward()
        optimizer.step()

        running_loss += loss.item()/num_batches #TOCHECK L1 norm?

        #Progress Bar
        progress_bar.set_description(f"Epoch {epoch}")
        progress_bar.set_postfix(loss=running_loss)
        progress_bar.update(1)

    return running_loss


"""chkpt_dir = './mae/mae_visualize_vit_base.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_base_patch16')"""
model_mae = models_mae.mae_vit_base_patch16_dec512d8b_custom()




trainset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=(224,224), sequence_length=1, norm_type="min_max")
trainset.train()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                        shuffle=True, num_workers=16,
                                        worker_init_fn=D.init_worker, drop_last=True)

valset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = False, img_size=(224,224), sequence_length=1, norm_type="min_max")
valset.eval()
valloader =  torch.utils.data.DataLoader(valset, batch_size=128,
                                    shuffle=False, num_workers=16,
                                    worker_init_fn=D.init_worker, drop_last=True)


optimizer = torch.optim.Adam(model_mae.parameters(), 5e-3)
model_mae = model_mae.to(device)
model_mae = train(model_mae, trainloader, optimizer, device, 100)

torch.save(model_mae.state_dict(), "./mae_base_from_scratch.pt")




