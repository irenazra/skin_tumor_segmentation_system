# File that establishes the U-Net architecture

# Import depedencies
import torch
from torch import Tensor
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule

# Convolution class using RELU activation
class ConvNormRelu(nn.Module):
    def __init__(self, in_features: int, out_features: int):

        super(ConvNormRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3),
            nn.BatchNorm2d(out_features),
            nn.ReLU())
        self.in_features = in_features
        self.out_features = out_features
        
    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x) 

# Class for downstream section of the U-Net
class UnetDown (nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvNormRelu(in_channels,out_channels)
        self.conv2 = ConvNormRelu(out_channels,out_channels)
        self.rpad = nn.ReplicationPad2d(1)
        self.pool = nn.MaxPool2d(2)
        

    def forward(self,x:Tensor) -> Tensor:
        c1 = self.conv1(x)
        p1 = self.rpad(c1)
        c2 = self.conv2(p1)
        p2 = self.rpad(c2)

        return self.pool(p2), p2

# Class for upstream section of the U-Net
class UnetUp(nn.Module):
    def __init__(self, in_channels, mid_channels,out_channels) :
        super().__init__()
        self.conv1 = ConvNormRelu(in_channels,mid_channels)
        self.conv2 = ConvNormRelu(mid_channels,mid_channels)
        self.rpad = nn.ReplicationPad2d(1)
        self.trans = nn.ConvTranspose2d(mid_channels, out_channels, 2, stride=2)
    

    def forward(self,x:Tensor) -> Tensor:
        c1 = self.conv1(x)
        p1 = self.rpad(c1)
        c2 = self.conv2(p1)
        p2 = self.rpad(c2)
        t = self.trans(p2)

        return t

# Class for final output of the U-Net
class UnetOut(nn.Module):
    # out_channels here is the number of classes
    def __init__(self, in_channels,mid_channels,out_channels) :
        super().__init__()
        self.conv1 = ConvNormRelu(in_channels,mid_channels)
        self.conv2 = ConvNormRelu(mid_channels,mid_channels)
        self.rpad = nn.ReplicationPad2d(1)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self,x:Tensor) -> Tensor:
        c1 = self.conv1(x)
        p1 = self.rpad(c1)
        c2 = self.conv2(p1)
        p2 = self.rpad(c2)
        result = self.conv3(p2)

        return result

# Class that builds entire Unet
class Unet(nn.Module):
    def __init__(self, num_in_channels: int = 3, num_out_channels: int = 3, max_features: int = 1024):
   
        super(Unet, self).__init__()

        features_4 = max_features // 2
        features_3 = features_4 // 2
        features_2 = features_3 // 2
        features_1 = features_2 // 2

        # Downstream section - 4 conv. blocks
        self.conv_block1 = UnetDown(num_in_channels, features_1)
        self.conv_block2 = UnetDown(features_1, features_2)
        self.conv_block3 = UnetDown(features_2, features_3)
        self.conv_block4 = UnetDown(features_3, features_4)

        # Upstream  section - 4 conv. blocks
        self.deconv_block1 = UnetUp(features_4, max_features, features_4)
        self.deconv_block2 = UnetUp(max_features, features_4, features_3)
        self.deconv_block3 = UnetUp(features_4, features_3, features_2)
        self.deconv_block4 = UnetUp(features_3, features_2, features_1)

        # Output section
        self.output_block = UnetOut(features_2, int(features_2/2), num_out_channels)

    # Establish how to make a forward pass
    def forward(self, x: Tensor) -> Tensor:
    
        # Pass and modify x donwstream
        x, c1 = self.conv_block1(x)
        x, c2 = self.conv_block2(x)
        x, c3 = self.conv_block3(x)
        x, c4 = self.conv_block4(x)

        # Pass and modify x upstream
        d1 = self.deconv_block1(x)
        d2 = self.deconv_block2(torch.cat((c4, d1), dim=1))
        d3 = self.deconv_block3(torch.cat((c3, d2), dim=1))
        d4 = self.deconv_block4(torch.cat((c2, d3), dim=1))

        # Finalize output and return
        out = self.output_block(torch.cat((c1, d4), dim=1))
        return out





       

