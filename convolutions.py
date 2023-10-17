import torch
import torch.nn as nn

"""
Classic 2d Convolutions blocks
"""


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.actv = nn.ReLU()
        #self.actv = nn.Tanh()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.actv(x)

        return x
    
class ConvTranspBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv_tr1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.conv_tr2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=1, stride=stride)
        #self.actv = nn.Tanh()
        self.actv = nn.ReLU()
    
    def forward(self, x):
        x = self.actv(x)
        x = self.conv_tr1(x)
        x = self.actv(x)
        x = self.conv_tr2(x)

        return x
    
class TemporalExpansion(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        """
        2d convolutions with kernel size only the expand the number of channels
        """
        self.conv1 = nn.Conv2d(sequence_length, 128, 1)
        self.conv2 = nn.Conv2d(128, 256, 1)
        self.actv = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.actv(x)
        return x
    
"""
Temporal Convolutional Networks; These 
"""


class TCN_v1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, dilation = 2**depth, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, dilation = 2**depth, stride=stride)
        self.actv = nn.ReLU()
    
    def forward(self, input_sequence):
        x = self.conv1(input_sequence)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    
class TCN_tr_v1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth, stride=1):
        super().__init__()
        self.conv_tr1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=1, dilation = 2**depth, stride=stride)
        self.conv_tr2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size, padding=1, dilation = 2**depth, stride=stride)
        self.actv = nn.ReLU()
    
    def forward(self, input_sequence):
        x = self.conv_tr1(input_sequence)
        x = self.actv(x)
        x = self.conv_tr2(x)
        return x
    
"""
This is for the autoencoder with TCNs. Image size (424, 512)


"""

"""
Module to expand th channels before the dilated 3d convolutions
"""
class TemporalExpansion3d(nn.Module):
    def __init__(self, sequence_length):
        super().__init__()
        """
        2d convolutions with kernel size only the expand the number of channels
        """
        self.conv1 = nn.Conv3d(sequence_length, 128, 1)
        self.conv2 = nn.Conv3d(128, 256, 1)
        self.actv = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        x = self.actv(x)
        return x


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=1, stride=stride)
        #self.actv = nn.ReLU()
        self.actv = nn.Tanh()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x
    
class Conv3dTranspBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv_tr1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.conv_tr2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size, padding=1, stride=stride)
        self.actv = nn.Tanh()
    
    def forward(self, x):
        x = self.conv_tr1(x)
        x = self.actv(x)
        x = self.conv_tr2(x)
        return x