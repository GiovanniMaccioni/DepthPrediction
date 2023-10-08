import torch
import models as M
import data as D


#Hyperparameters


#The dimension of images is that of depth images in the SimBa dataset
#x = torch.randn(1, 1, 10, 424, 512)#[batch_size, channels, sequence_len, height, width]
x = torch.randn(1, 10*1, 424, 512)#[batch_size, sequence_len*channels, height, width]

dataset =  D.BaxterJointsSynthDataset("./data/dataset", [0], "train", demo = True, img_size=(512, 424), sequence_length=5)
dataset.train()


trainloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                        shuffle=True, num_workers=4,
                                        worker_init_fn=D.init_worker, drop_last=True),

dataset.eval()
valloader =  torch.utils.data.DataLoader(dataset, batch_size=4,
                                    shuffle=False, num_workers=4,
                                    worker_init_fn=D.init_worker, drop_last=True)


#TOCHECK I don't know if I have to do the XYZ transformation as said in the paper
encoder = M.Encoder()
decoder = M.Decoder()
model = M.Autoencoder(encoder, decoder)

out, latent_vector = model(x)

print("ciao")

"""
- It works also with image dimension 424, 424. For example we can crop the image as the robot seems to be always at the center
    of the image----> Now changing the size of the image leads to the modification of the convolutional layers!!!

- I have to see if passing the sequnce of images like this (concatenate them along the channels), is valid or not

- The architecture as it is has over 28M parameters. We can think of reduce the dimension of the images
     with a convolution overhead.(And then adjust the number of convolutional layers)

- The latent vector is very big so the reduction previuosly mention chìan favour the reduction of its dimensoion also; as another
    solution we can add other layers to reduce directly the dimension(Idea proper conv1d after the 2d cov with kernel size = 1)

- To add wandb to track the experiment
"""
