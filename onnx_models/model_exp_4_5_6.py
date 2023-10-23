import torch
import torch.nn as nn
from convolutions import *


"""
FIXME
Autoencoder that use Temporal Convolutional Neural Networks to embed the sequence of frames and 
Temporal Convolutional Networks to produce the current and next frames
"""

#Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        #For the first convolution, we have input channels = 1. 32 and 3 are chosen arbitrarly for now
        self.conv1 = ConvBlock(1*1, 8, 3)#The input channels depend on the sequence length chosen
        self.conv2 = ConvBlock(8, 16, 3)
        self.conv3 = ConvBlock(16, 32, 3)
        self.conv4 = ConvBlock(32, 64, 3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        """x = self.tcn5(x)
        x = self.tcn6(x)"""

        #x = self.pool(x)
        return x
        
#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_tr1 = ConvTranspBlock(64, 32, 3)
        self.conv_tr2 = ConvTranspBlock(32, 16, 3)
        self.conv_tr3 = ConvTranspBlock(16, 8, 3)
        self.conv_tr4 = ConvTranspBlock(8, 1, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage 
                                            # it can be the last frame or the next one
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
    def forward(self, x):
        x = self.upsample(x)

        x = self.conv_tr1(x)
        x = self.conv_tr2(x)

        x = self.upsample(x)

        x = self.conv_tr3(x)
        x = self.conv_tr4(x)
        return x

#Autoencoder
class Autoencoder_conv(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        """self.conv2d1 = nn.Conv2d(256, 256, kernel_size = [2, 3], stride = [2, 2], padding = [7, 1])
        self.conv2d2 = nn.Conv2d(256, 256, kernel_size = [3, 3], stride = [2, 2], padding = [3, 1])
        self.conv1d = nn.Conv2d(256, 1, 1)
        self.conv1d_tr = nn.ConvTranspose2d(1, 256, 1)
        self.conv2d_tr1 = nn.ConvTranspose2d(256, 256, kernel_size = [4, 4], stride = [2, 2], padding = [3, 1])
        self.conv2d_tr2 = nn.ConvTranspose2d(256, 256, kernel_size = [2, 4], stride = [2, 2], padding = [7, 1])"""

        #self.actv = nn.Tanh()
        self.actv = nn.ReLU()

    def encode(self, input_sequence):
        return self.enc(input_sequence)
    
    def decode(self, latent_vector):
        return self.dec(latent_vector)
    
    def forward(self, input_sequence):
        x = self.encode(input_sequence)
        """###
        x = self.conv2d1(x)
        x = self.actv(x)
        x = self.conv2d2(x)
        x = self.actv(x)
        x = self.conv1d(x)
        x = self.actv(x)
        height, width = x.shape[2], x.shape[3] 
        latent_vector = x.flatten(1)
        x = latent_vector.unflatten(1, torch.Size([1, height, width]))
        x = self.conv1d_tr(x)
        x = self.actv(x)
        x = self.conv2d_tr1(x)
        x = self.actv(x)
        x = self.conv2d_tr2(x)
        ###"""
        out = self.decode(x)
        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, None#latent_vector
 
    
    
#Encoder
class Encoder_3d(nn.Module):
    def __init__(self):
        super().__init__()
        #For the first convolution, we have input channels = 1. 32 and 3 are chosen arbitrarly for now
        self.cn3d1 = Conv3dBlock(256, 32, 3)#The input channels depend on the sequence length chosen
        self.cn3d2 = Conv3dBlock(32, 64, 3)
        self.cn3d3 = Conv3dBlock(64, 128, 3)
        self.cn3d4 = Conv3dBlock(128, 256, 3)
        """self.tcn5 = TCN_v1(256, 512, 3, 4)
        self.tcn6 = TCN_v1(512, 1024, 3, 5)"""
        self.pool = nn.MaxPool3d(3, stride=2, padding=1)
    
    def forward(self, input_sequence):
        x = self.cn3d1(input_sequence)
        x = self.cn3d2(x)
        x = self.pool(x)
        x = self.cn3d3(x)

        #x = self.pool(x)

        x = self.cn3d4(x)
        x = self.pool(x)
        """x = self.tcn5(x)
        x = self.tcn6(x)"""

        #x = self.pool(x)
        return x
        
#Decoder
class Decoder_3d(nn.Module):
    def __init__(self):
        super().__init__()
        """self.tcn1_tr = TCN_tr_v1(1024, 512, 3, 5)
        self.tcn2_tr = TCN_tr_v1(512, 256, 3, 4)"""
        self.cn3d1_tr = Conv3dTranspBlock(256, 128, 3)
        self.cn3d2_tr = Conv3dTranspBlock(128, 64, 3)
        self.cn3d3_tr = Conv3dTranspBlock(64, 32, 3)
        self.cn3d4_tr = Conv3dTranspBlock(32, 1, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage 
                                            # it can be the last frame or the next one
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        
    def forward(self, latent_vector):
        x = self.upsample(latent_vector)

        x = self.cn3d1_tr(x)
        x = self.cn3d2_tr(x)

        x = self.upsample(x)

        x = self.cn3d3_tr(x)
        x = self.cn3d4_tr(x)

        """x = self.tcn5_tr(x)
        x = self.tcn6_tr(x)"""
        
        #x = self.upsample(x)
        return x   

#Autoencoder
class Autoencoder_3d(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        self.temp_expansion = TemporalExpansion(1)

        """self.conv2d1 = nn.Conv3d(256, 256, kernel_size = [1, 2, 3], stride = [1, 2, 2], padding = [0, 7, 1])
        self.conv2d2 = nn.Conv3d(256, 256, kernel_size = [1, 3, 3], stride = [1, 2, 2], padding = [0, 3, 1])
        self.conv1d = nn.Conv3d(256, 1, 1)
        self.conv1d_tr = nn.ConvTranspose3d(1, 256, 1)
        self.conv2d_tr1 = nn.ConvTranspose3d(256, 256, kernel_size = [1, 4, 4], stride = [1, 2, 2], padding = [0, 3, 1])
        self.conv2d_tr2 = nn.ConvTranspose3d(256, 256, kernel_size = [1, 2, 4], stride = [1, 2, 2], padding = [0, 7, 1])"""

        self.actv = nn.Tanh()
    
    def encode(self, input_sequence):
        return self.enc(input_sequence)
    
    def decode(self, latent_vector):
        return self.dec(latent_vector)
    
    def forward(self, x):
        x = self.temp_expansion(x)
        x = self.encode(x)
        """###
        x = self.conv2d1(x)
        x = self.actv(x)
        x = self.conv2d2(x)
        x = self.actv(x)
        x = self.conv1d(x)
        x = self.actv(x)
        channels, height, width = x.shape[2], x.shape[3], x.shape[4] 
        latent_vector = x.flatten(1)
        x = latent_vector.unflatten(1, torch.Size([1, channels, height, width]))
        x = self.conv1d_tr(x)
        x = self.actv(x)
        x = self.conv2d_tr1(x)
        x = self.actv(x)
        x = self.conv2d_tr2(x)
        ###"""
        out = self.decode(x)
        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, None#latent_vector