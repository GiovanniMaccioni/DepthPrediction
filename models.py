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
        self.pre_conv1 = nn.Conv2d(1*1, 4, 7, padding=1, stride=2)
        self.conv1 = ConvBlockResNetFirst(4, 8, 3)#The input channels depend on the sequence length chosen

        self.pre_conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        self.conv2 = ConvBlockResNet(8, 16, 3)

        self.pre_conv3 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.conv3 = ConvBlockResNet(16, 32, 3)

        self.pre_conv4 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv4 = ConvBlockResNet(32, 64, 3)

        self.pre_conv5 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv5 = ConvBlockResNet(64, 128, 3)

        self.pre_conv6 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv6 = ConvBlockResNet(128, 256, 3)

        self.pre_conv7 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv7 = ConvBlockResNet(256, 512, 3)

        self.pre_conv8 = nn.Conv2d(256, 512, 3, padding=1, stride=2)
        self.conv8 = ConvBlockResNetLast(512, 3)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.actv = nn.ReLU()
    
    def forward(self, x):
        x = self.pre_conv1(x)
        x = self.actv(x)
        x = self.pool(x)
        x, proj = self.conv1(x)

        x = self.pre_conv2(x)
        x = self.actv(x)
        x, proj = self.conv2(x, proj)

        x = self.pre_conv3(x)
        x = self.actv(x)
        x, proj = self.conv3(x, proj)

        x = self.pre_conv4(x)
        x = self.actv(x)
        x, proj = self.conv4(x, proj)

        x = self.pre_conv5(x)
        x = self.actv(x)
        x, proj = self.conv5(x, proj)

        x = self.pre_conv6(x)
        x = self.actv(x)
        x, proj = self.conv6(x, proj)

        x = self.pre_conv7(x)
        x = self.actv(x)
        x, proj = self.conv7(x, proj)

        x = self.pre_conv8(x)
        x = self.actv(x)
        x = self.conv8(x, proj)

        return x
        
#Decoder
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        """
        self.pre_conv1 = nn.Conv2d(1*1, 4, 7, padding=1, stride=2)
        self.conv1 = ConvBlockResNetFirst(4, 8, 3)#The input channels depend on the sequence length chosen

        self.pre_conv2 = nn.Conv2d(4, 8, 3, padding=1, stride=2)
        self.conv2 = ConvBlockResNet(8, 16, 3)

        self.pre_conv3 = nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.conv3 = ConvBlockResNet(16, 32, 3)

        self.pre_conv4 = nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv4 = ConvBlockResNetLast(32, 3)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.actv = nn.Tanh()
        """
        #The Resnet blocks (except for the last) will have a double output, the feature map and the projection
        #the second argument is refered to the number of out_channels of the projection
        self.conv_tr1 = ConvTranspBlockResNetFirst(2048, 256, 3, output_padding=(1,1))
        self.post_conv1 = nn.ConvTranspose2d(2048, 256, 3, padding=1, stride=2, output_padding=(1,1))#kernel_size = (3,4)

        self.conv_tr2 = ConvTranspBlockResNet(256, 128, 3, output_padding=(1,1))
        self.post_conv2 = nn.ConvTranspose2d(256, 128, 3, padding=1, stride=2, output_padding=(1,1))#kernel_size = (3,4)

        self.conv_tr3 = ConvTranspBlockResNet(128, 64, 3, output_padding=(0,1))
        self.post_conv3 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=(0,1))#kernel_size = (3,4)

        self.conv_tr4 = ConvTranspBlockResNet(64, 32, 3, output_padding=(1,1))
        self.post_conv4 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=(1,1))#kernel_size = (3,4)

        self.conv_tr5 = ConvTranspBlockResNet(32, 16, 3, output_padding=(0,1))
        self.post_conv5 = nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=(0,1))

        self.conv_tr6 = ConvTranspBlockResNet(16, 8, 3, output_padding=(1,1))
        self.post_conv6 = nn.ConvTranspose2d(16, 8, 3, padding=1, stride=2, output_padding=(1,1))

        self.conv_tr7 = ConvTranspBlockResNet(8, 4, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv7 = nn.ConvTranspose2d(8, 4, 3, padding=1, stride=2)

        self.conv_tr8 = ConvTranspBlockResNetLast(4, 1, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv8 = nn.ConvTranspose2d(4, 1, 7, padding=1, stride=2, output_padding=(1,1))
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        
    def forward(self, x):

        x, proj = self.conv_tr1(x)
        #x = self.actv(x)
        x = self.post_conv1(x)

        x, proj = self.conv_tr2(x, proj)
        #x = self.actv(x)
        x = self.post_conv2(x)

        x, proj = self.conv_tr3(x, proj)
        #x = self.actv(x)
        x = self.post_conv3(x)

        x, proj = self.conv_tr4(x, proj)
        #x = self.actv(x)
        x = self.post_conv4(x)

        x, proj = self.conv_tr5(x, proj)
        #x = self.actv(x)
        x = self.post_conv5(x)

        x, proj = self.conv_tr6(x, proj)
        #x = self.actv(x)
        x = self.post_conv6(x)

        x, proj = self.conv_tr7(x, proj)
        #x = self.actv(x)
        x = self.post_conv7(x)
        
        x = self.conv_tr8(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)#TODO Try with convolution
        x = self.post_conv8(x)

        return x
    
#Decoder
class Decoder_resconv(nn.Module):
    def __init__(self):
        super().__init__()
        #The Resnet blocks (except for the last) will have a double output, the feature map and the projection
        #the second argument is refered to the number of out_channels of the projection
        self.conv1 = ConvUPCONVBlockResNetFirst(512, 256, 3)
        self.post_conv1 = nn.Conv2d(512, 256, 3, padding=1, stride=1)#kernel_size = (3,4)

        self.conv2 = ConvUPCONVBlockResNet(256, 128, 3)
        self.post_conv2 = nn.Conv2d(256, 128, 3, padding=1, stride=1)#kernel_size = (3,4)

        self.conv3 = ConvUPCONVBlockResNet(128, 64, 3)
        self.post_conv3 = nn.Conv2d(128, 64, 3, padding=1, stride=1)#kernel_size = (3,4)

        self.conv4 = ConvUPCONVBlockResNet(64, 32, 3)
        self.post_conv4 = nn.Conv2d(64, 32, 3, padding=1, stride=1)#kernel_size = (3,4)

        self.conv5 = ConvUPCONVBlockResNet(32, 16, 3)
        self.post_conv5 = nn.Conv2d(32, 16, 3, padding=1, stride=1)

        self.conv6 = ConvUPCONVBlockResNet(16, 8, 3)
        self.post_conv6 = nn.Conv2d(16, 8, 3, padding=1, stride=1)

        self.conv7 = ConvUPCONVBlockResNet(8, 4, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv7 = nn.Conv2d(8, 4, 3, padding=1, stride=1)

        self.conv8 = ConvUPCONVBlockResNetLast(4, 1, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv8 = nn.Conv2d(4, 1, 3, padding=1, stride=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample432x512 = nn.Upsample(size=(432, 512), mode="nearest")

        
    def forward(self, x):

        x, proj = self.conv1(x)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv1(x)

        x, proj = self.conv2(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv2(x)

        x, proj = self.conv3(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv3(x)

        x, proj = self.conv4(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv4(x)

        x, proj = self.conv5(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv5(x)

        x, proj = self.conv6(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv6(x)

        x, proj = self.conv7(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)
        x = self.post_conv7(x)
        
        x = self.conv8(x, proj)
        #x = self.actv(x)
        #x = self.upsample(x)#TODO Try with convolution
        x = self.upsample432x512(x)#TODO Try with convolution

        x = self.post_conv8(x)

        return x

#Autoencoder
class Autoencoder_conv(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        #TODO add Flatten and Unflatten
        """self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(2048, 256)#FIXME 512 instead of 1000(resnet152) for my encoder
        self.linear2 = nn.Linear(256, 128)

        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 2048)"""
        
        #self.actv = nn.Tanh()
        self.actv = nn.ReLU()
        self.actv_out = nn.Sigmoid()


        #TOCHECK  for SSIM Loss
        #self.actv_out = nn.Sigmoid()

    def encode(self, input_sequence):
        return self.enc(input_sequence)
    
    def decode(self, latent_vector):
        return self.dec(latent_vector)
    
    def forward(self, input_sequence):
        x = self.encode(input_sequence)

        """#x = self.flatten(x)
        x = self.linear1(x)#FIXME x.permute(0,2,3,1) for my encoder
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)

        x = self.linear3(x)
        x = self.actv(x)
        x = self.linear4(x)
        #x = x.reshape((x.shape[0],128,4,6))"""
        x = x[:, :, None, None]#FIXME for my encoder x.permute(0,3,1,2)
        
        out = self.decode(x)

        out = self.actv_out(out)

        #TOCHECK  for SSIM Loss
        #out = self.actv_out(out)


        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, None##latent_vector
    
#Autoencoder
class VAE_conv(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        #TODO add Flatten and Unflatten
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)

        self.linear3 = nn.Linear(128, 256)
        self.linear4 = nn.Linear(256, 512)
        
        #self.actv = nn.Tanh()
        self.actv = nn.ReLU()

        #TOCHECK  for SSIM Loss
        #self.actv_out = nn.Sigmoid()

    def encode(self, x):

        return self.enc(x)
    
    def decode(self, latent_vector):

        return self.dec(latent_vector)
    
    def forward(self, input_sequence):
        x = self.encode(input_sequence)

        #x = self.flatten(x)
        x = self.linear1(x.permute(0,2,3,1))
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)

        x = self.linear3(x)
        x = self.actv(x)
        x = self.linear4(x)
        #x = x.reshape((x.shape[0],128,4,6))
        x = x.permute(0,3,1,2)
        
        out = self.decode(x)

        #TOCHECK  for SSIM Loss
        #out = self.actv_out(out)


        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, None#None#latent_vector
    
    def sample(self):
        return
    
    def reparametrize(self):
        return
    

    
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
    

class Autoencoder_MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256*256, 2048)
        self.linear2 = nn.Linear(2048, 1024)
        self.linear3 = nn.Linear(1024, 256)
        self.linear4 = nn.Linear(256, 128)

        self.linear5 = nn.Linear(128, 256)
        self.linear6 = nn.Linear(256, 1024)
        self.linear7 = nn.Linear(1024, 2048)
        self.linear8 = nn.Linear(2048, 256*256)

        self.actv = nn.LeakyReLU(0.2)


    def forward(self, x):

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)
        x = self.linear3(x)
        x = self.actv(x)
        x = self.linear4(x)

        x = self.actv(x)
        x = self.linear5(x)
        x = self.actv(x)
        x = self.linear6(x)
        x = self.actv(x)
        x = self.linear7(x)
        x = self.actv(x)
        x = self.linear8(x)

        x = x.reshape((x.shape[0],1,256,256))

        return x, None

