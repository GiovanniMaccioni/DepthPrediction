import model as M
import data as D

import torch
import os
from sklearn import manifold
import matplotlib.pyplot as plt

"""# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401"""
from matplotlib import ticker

def plot_2d(points, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points)
    plt.show()

def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

t_sne = manifold.TSNE(
    n_components=2,
    perplexity=30,
    init="random",
    n_iter=250,
    random_state=0,
)

encoder = M.Encoder()
decoder = M.Decoder()
autoencoder = M.Autoencoder_conv(encoder, decoder)
autoencoder.load_state_dict(torch.load(os.path.join("./runs_augm/run25", "model.pt")))

valset =  D.RecDataset("./data/dataset", "train", img_size=(256, 256), sequence_length=1)
valset.eval()
valloader =  torch.utils.data.DataLoader(valset, batch_size=1,
                                    shuffle=False, num_workers=1,
                                    worker_init_fn=D.init_worker, drop_last=True)
latents = []
with torch.no_grad():
    for i, images in enumerate(valloader):
        images = images["depth8"]

        latent = autoencoder.encode(images)

        latents.append(latent)
    

S_t_sne = t_sne.fit_transform(latents)

plot_2d(S_t_sne, "T-distributed Stochastic  \n Neighbor Embedding")