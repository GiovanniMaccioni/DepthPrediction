import torch
import numpy as np
import random

import matplotlib.pyplot as plt


def show_depths(depths, colormap = 'magma'):
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
    return
