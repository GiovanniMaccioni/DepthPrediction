"""
FIXME
Autoencoder that use Temporal Convolutional Neural Networks to embed the sequence of frames and 
Temporal Convolutional Networks to produce the current and next frames
"""


from convolutions import *

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
        self.conv6 = ConvBlockResNetLast(128, 3)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.actv = nn.Tanh()
    
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
        x = self.conv6(x, proj)

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
        self.conv_tr1 = ConvTranspBlockResNetFirst(128, 64, 3, output_padding=(0,1))
        self.post_conv1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2, output_padding=(0,1))#kernel_size = (3,4)

        self.conv_tr2 = ConvTranspBlockResNet(64, 32, 3, output_padding=(1,1))
        self.post_conv2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=(1,1))#kernel_size = (3,4)

        self.conv_tr3 = ConvTranspBlockResNet(32, 16, 3, output_padding=(0,1))
        self.post_conv3 = nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2, output_padding=(0,1))

        self.conv_tr4 = ConvTranspBlockResNet(16, 8, 3, output_padding=(1,1))
        self.post_conv4 = nn.ConvTranspose2d(16, 8, 3, padding=1, stride=2, output_padding=(1,1))

        self.conv_tr5 = ConvTranspBlockResNet(8, 4, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv5 = nn.ConvTranspose2d(8, 4, 3, padding=1, stride=2)

        self.conv_tr6 = ConvTranspBlockResNetLast(4, 1, 3)#TOCHECK The out_channels is at 1 because we have to produce an image; at this stage
        self.post_conv6 = nn.ConvTranspose2d(4, 1, 7, padding=1, stride=2, output_padding=(1,1))
        
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.actv = nn.Tanh()
        
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
        
        x = self.conv_tr6(x, proj)
        #x = self.actv(x)
        x = self.upsample(x)#TODO Try with convolution
        x = self.post_conv6(x)

        return x

#Autoencoder
class Autoencoder_conv(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

        #TODO add Flatten and Unflatten
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(3072, 1024)
        self.linear2 = nn.Linear(1024, 256)

        self.linear3 = nn.Linear(256, 1024)
        self.linear4 = nn.Linear(1024, 3072)
        
        #self.actv = nn.Tanh()
        self.actv = nn.LeakyReLU(0.1)

    def encode(self, input_sequence):
        return self.enc(input_sequence)
    
    def decode(self, latent_vector):
        return self.dec(latent_vector)
    
    def forward(self, input_sequence):
        x = self.encode(input_sequence)

        x = self.flatten(x)
        x = self.linear1(x)
        x = self.actv(x)
        x = self.linear2(x)
        x = self.actv(x)

        x = self.linear3(x)
        x = self.actv(x)
        x = self.linear4(x)
        x = x.reshape((x.shape[0],128,4,6))
        
        out = self.decode(x) 
        #Added the ReLU activation to have only positive values in the output
        #as we are dealing with the depth values
        return out, x#None#latent_vector
    

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