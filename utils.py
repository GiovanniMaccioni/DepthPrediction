import torch
import numpy as np
import random

import matplotlib.pyplot as plt

import os

import pandas as pd


def show_depths(depths, colormap = 'magma'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    # Plot each image in the corresponding position
    axs[0].imshow(depths[0][0].cpu().detach().numpy(), cmap=colormap)
    axs[0].set_title('Original Depth')

    axs[1].imshow(depths[1][0].cpu().detach().numpy(), cmap=colormap)
    axs[1].set_title('Reconstructed Depth')

    plt.show()
    plt.close(fig)

def save_depths(depths, colormap = 'magma'):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))
    
    # Plot each image in the corresponding position
    axs[0].imshow(depths[0][0].cpu().detach().numpy(), cmap=colormap)
    axs[0].set_title('Original Depth')

    axs[1].imshow(depths[1][0].cpu().detach().numpy(), cmap=colormap)
    axs[1].set_title('Reconstructed Depth')

    plt.show()
    plt.close(fig)

def set_reproducibility(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    #torch.backends.cudnn.deterministic = True
    
    return

def show_image(image, title=''):
    # image is [H, W, 3]
    #plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.imshow(image.cpu().detach().numpy())
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.show()
    plt.close()
    return

"""def run_one_image(img, model, mask_ratio=0.75):
    x = torch.tensor(img).to(device)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
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

    plt.show()"""

def save_depths_sequence(depths, destination_path, minus=False):
    fig, axs = plt.subplots(1, depths.shape[0], figsize=(20, 20))

    # Plot each image in the corresponding position
    for i in range(depths.shape[0]):
        axs[i].imshow(depths[i][0].cpu().detach().numpy())
        axs[i].axis("off")
        axs[i].set_title(f"{i-1 if minus else i}")

    plt.savefig(destination_path, bbox_inches='tight')
    plt.close(fig)
    return

def visualize_depths_sequence(depths_list, sequence_length, colormap = 'magma'):
    fig, axs = plt.subplots(1, sequence_length, figsize=(10, 10))

    # Plot each image in the corresponding position
    for i in range(sequence_length):
        axs[i].imshow(depths_list[i][0].cpu().detach().numpy(), cmap=colormap)
        
        #axs[0].set_title('Original Depth')

    plt.show()
    plt.close(fig)
    return

def save_run_results(model, dir_path, loader, device, config):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))
    with open(os.path.join(dir_path, "config.txt"), 'w') as f:
        print(config, file=f)

    with torch.no_grad():
        for i, images in enumerate(loader):

            images = images["depth8"].to(device)
            out, _ = model(images[0])

            save_depths_sequence(images[0], os.path.join(dir_path, f"val_seq_gt_{i}.png"))
            save_depths_sequence(out, os.path.join(dir_path, f"val_seq_prediction_{i}.png"))

            if i == 9:
                break

    return

def save_run_results_lstm(lstm, autoencoder, dir_path, loader, device, config):
    autoencoder.eval()
    lstm.eval()

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    torch.save(lstm.state_dict(), os.path.join(dir_path, "model.pt"))
    with open(os.path.join(dir_path, "config.txt"), 'w') as f:
        print(config, file=f)

    with torch.no_grad():
        for i, images in enumerate(loader):
            images = images["depth8"].to(device)

            batch_len = images.shape[0]
            seq_len = images.shape[1]
            images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3], images.shape[4])
            latent = autoencoder.encode(images)
            images = images.reshape(batch_len, seq_len, images.shape[1], images.shape[2], images.shape[3])
            new_batch = latent.reshape(batch_len,seq_len, latent.shape[1]*latent.shape[2]*latent.shape[3])

            output, _, _ = lstm(new_batch[:,:-1, :])#one step prediction
            
            output = output.reshape(output.shape[0]*output.shape[1], 512, 1, 1)
            pred_images = autoencoder.decode(output)
            pred_images = pred_images.reshape(batch_len, seq_len-1, pred_images.shape[1], pred_images.shape[2], pred_images.shape[3])

            #RMSE between the produced image and the gt at the corresponding time step
            rmse1 = torch.sqrt(torch.mean((pred_images - images[:, 1:, :])**2, dim=(3,4))).squeeze()
            #RMSE between the produced image and the gt at the previous time step
            rmse2 = torch.sqrt(torch.mean((pred_images - images[:, :-1, :])**2, dim=(3,4))).squeeze()
            #RMSE between the previous gt and the gt at the next timestep
            rmse3 = torch.sqrt(torch.mean((images[:, :-1, :] - images[:, 1:, :])**2, dim=(3,4))).squeeze()

            df = pd.DataFrame()

            df.insert(0, "rmse t_gt t-1_gt", rmse3.cpu().numpy())
            df.insert(0, "rmse t-1_gt t_pred", rmse2.cpu().numpy())
            df.insert(0, "rmse t_gt t_pred", rmse1.cpu().numpy())
            
            df.to_csv(os.path.join(dir_path, f"val_seq_prediction_{i}.csv"))

            save_depths_sequence(images[0], os.path.join(dir_path, f"val_seq_gt_{i}.png"), True)
            save_depths_sequence(pred_images[0], os.path.join(dir_path, f"val_seq_prediction_{i}.png"))

            if i == 9:
                break
    return
    