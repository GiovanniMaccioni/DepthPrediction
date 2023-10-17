import torch
import numpy as np
import random

import matplotlib.pyplot as plt


def show_depths(depths):
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Choose the desired colormap
    colormap = 'magma'
    
    # Plot each image in the corresponding position
    axs[0].imshow(depths[0][0].cpu().detach().numpy(), cmap='magma')
    axs[0].set_title('Original Depth')

    axs[1].imshow(depths[1][0].cpu().detach().numpy(), cmap='magma')
    axs[1].set_title('Reconstructed Depth')

    plt.show()
    plt.close(fig)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.use_deterministic_algorithms(True)
    return
