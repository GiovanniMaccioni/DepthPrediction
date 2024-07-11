import torch
import torch.nn as nn


"""
FIXME
Autoencoder that use Temporal Convolutional Neural Networks to embed the sequence of frames and 
Temporal Convolutional Networks to produce the current and next frames
"""
class ConvUpResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)

        return x

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        """#For the first convolution, we have input channels = 1. 32 and 3 are chosen arbitrarly for now
        self.conv1 = nn.Conv2d(1*1, 4, 3, 1, 1)#The input channels depend on the sequence length chosen
        self.conv2 = nn.Conv2d(4, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        #self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)"""

        """
        ---->run4 + pool"""
        self.conv1 = nn.Conv2d(1*1, 4, 3, 1, 1)#The input channels depend on the sequence length chosen
        self.conv2 = nn.Conv2d(4, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.fin_pool1 = nn.MaxPool2d(4, stride=1, padding=0)
        #self.fin_pool2 = nn.AvgPool2d(4, stride=1, padding=0)

        """
        ---->run4
        self.conv1 = nn.Conv2d(1*1, 4, 3, 1, 1)#The input channels depend on the sequence length chosen
        self.conv2 = nn.Conv2d(4, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)"""

        """self.conv1 = nn.Conv2d(1*1, 4, 3, 1, 1)#The input channels depend on the sequence length chosen
        self.conv2 = nn.Conv2d(4, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 1024, 3, 1, 1)"""

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        #self.actv = nn.ReLU()
        self.actv = nn.ELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.actv(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.actv(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.actv(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.actv(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.actv(x)
        x = self.pool(x)

        x = self.conv7(x)
        x = self.actv(x)
        #x = self.pool(x)
        x = self.fin_pool1(x)
        #x2 = self.fin_pool2(x)
        """x = self.conv8(x)
        x = self.actv(x)
        x = self.pool(x)"""
     
        #out = torch.cat((x1, x2), dim=1)

        return x
        
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        """self.conv1 = nn.Conv2d(256, 256, 3, 1, 1)#1024 instead of
        self.conv2 = nn.Conv2d(256, 256, 3, 1, 1)#256
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)#256
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 4, 3, 1, 1)
        self.conv7 = nn.Conv2d(4, 1, 3, 1, 1)"""

        """
        run4 + first upsample scale 4 """
        #self.pre_conv = nn.Conv2d(1024, 512, 1, 1)
        self.first_upsample = nn.Upsample(scale_factor=4, mode="nearest")
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)#1024 instead of"""
        self.conv2 = nn.Conv2d(512, 256, 3, 1, 1)#256
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)#256
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 4, 3, 1, 1)
        self.conv7 = nn.Conv2d(4, 1, 3, 1, 1)

        """
        run4
        self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)#1024 instead of
        self.conv2 = nn.Conv2d(512, 256, 3, 1, 1)#256
        self.conv3 = nn.Conv2d(256, 128, 3, 1, 1)#256
        self.conv4 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv6 = nn.Conv2d(32, 4, 3, 1, 1)
        self.conv7 = nn.Conv2d(4, 1, 3, 1, 1)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage 
                                            # it can be the last frame or the next one"""
        """self.conv1 = nn.Conv2d(512, 512, 3, 1, 1)#256
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)#256
        self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)#256
        self.conv4 = nn.Conv2d(256, 128, 3, 1, 1)#256
        self.conv5 = nn.Conv2d(128, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 32, 3, 1, 1)
        self.conv7 = nn.Conv2d(32, 4, 3, 1, 1)
        self.conv8 = nn.Conv2d(4, 1, 3, 1, 1)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage 
                                            # it can be the last frame or the next one"""
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        #self.actv = nn.ReLU()
        self.actv = nn.ELU()
        
    def forward(self, x):

        """x = self.pre_conv(x)
        x = self.actv(x)"""

        x = self.first_upsample(x)
        #x = self.upsample(x)
        x = self.conv1(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv2(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv3(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv4(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv5(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv6(x)
        x = self.actv(x)

        x = self.upsample(x)
        x = self.conv7(x)
        #x = self.actv(x)

        """x = self.upsample(x)
        x = self.conv8(x)
        #x = self.actv(x)"""

        return x

#Autoencoder
class Autoencoder_conv(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        #self.actv = nn.ReLU()
        self.actv_out = nn.Sigmoid()

    def encode(self, input_sequence):
        return self.enc(input_sequence)
    
    def decode(self, latent_vector):
        dec = self.dec(latent_vector)
        return self.actv_out(dec)
    
    def forward(self, x):
        x = self.encode(x)
        out = self.decode(x)
        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, None#latent_vector