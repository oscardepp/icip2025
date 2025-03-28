import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from utils import make_layers
except:
    from .utils import make_layers


class UNet(nn.Module): 
     """ 
     A residual neural network. Use `make_layers` to quickly build the network. 
  
     Args: 
         channels: The number of channels in the data, which should match the in_channels 
                   of the first convolution in the CNN and the out_channels of the last 
                   convolution. 
     """ 
     def __init__(self, channels):
        super(UNet, self).__init__()
        self.encoder = make_layers(DownBlock,
                                   4,
                                   [channels, 64, 128, 256],
                                   [64, 128, 256, 512])#,
                                #    name='down')
        self.double_conv = DoubleConv(512, 512)
        self.decoder = make_layers(UpBlock,
                                   4,
                                   [512, 256, 128, 64],
                                   [512 + 512, 256 + 256, 128 + 128, 64 + 64],
                                   [256, 128, 64, 64])#,
                                #    name='up')
        self.final_conv = nn.Conv2d(64, channels, kernel_size=1)
        # add a final softplus activation function
        self.softplus = nn.Softplus()
     def forward(self, x):
        down_outs = []
        for layer in self.encoder:
            x = layer.double_conv(x)
            down_outs.append(x)
            x = layer.max_pool(x)
            
        x = self.double_conv(x)
        
        for layer, output in zip(self.decoder, down_outs[::-1]):
            x = layer.up_conv(x)
            x = layer.double_conv(torch.cat((x, output), dim = 1))
            
        x = self.final_conv(x)
        # added in a softplus to ensure the output is positive
        x = self.softplus(x)
        return x
  
class DoubleConv(nn.Module): 
     """ 
     A building block layer in the UNet. The order of modules should be as follows: 
         (0): Conv2d(in_channels = in_channels, out_channels = out_channels, *) 
         (1): BatchNorm2d 
         (2): ReLU 
         (3): Conv2d(in_channels = out_channels, out_channels = out_channels, *) 
         (4): BatchNorm2d 
         (5): ReLU 
  
     Args: 
         in_channels: Number of channels in the input image 
         out_channels: Number of channels produced by the convolution 
         args: Other arguments for the `nn.Conv2d` class 
         kwargs: Other keyword arguments for the `nn.Conv2d` class 
     """ 
     def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
     def forward(self, x):
        return self.double_conv(x)
  

class DownBlock(nn.Module): 
     """ 
     A building block layer in the UNet. The order of modules should be as follows: 
         (0): DoubleConv
         (1): MaxPool2d(kernel_size = 2, stride = 2) 
     MaxPool2d is listed first since we need to connect the prior output with a future 
     convolution. Otherwise, it would be more difficult to extract that output. 
  
     Args: 
         in_channels: Number of channels in the input image 
         out_channels: Number of channels produced by the convolution 
         args: Other arguments for the `DoubleConv` class 
         kwargs: Other keyword arguments for the `DoubleConv` class 
     """ 
     def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

     def forward(self, x):
        return self.max_pool(self.double_conv(x))

  
  
class UpBlock(nn.Module):
    """
    A building block layer in the UNet. The order of modules should be as follows:
        (0): ConvTranspose2d(kernel_size = 2, stride = 2)
        (1): DoubleConv
    ConvTranspose2d is listed second since we need to concatenate the output of the
    UpBlock with the output of a DownBlock.

    Args:
        in_channels: Number of channels in the input image
        out_channels: Number of channels produced by the convolution
        args: Other arguments for the `DoubleConv` class
        kwargs: Other keyword arguments for the `DoubleConv` class
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(UpBlock, self).__init__()
        
        # Upsample by a factor of 2
        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        # The DoubleConv will receive concatenated features, hence in_channels
        self.double_conv = DoubleConv(mid_channels, out_channels)
        

    def forward(self, x):
        
        return self.double_conv(self.up_conv(x))
#         x1 = self.up_conv(x1)
#         # Ensure dimensions for concatenation are correct
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
#         # Concatenate the upsampled output with the skip connection output
#         x = torch.cat([x2, x1], dim=1)  # Note the order of concatenation
#         print(f"Concatenated size: {x.size()}")  # Print the size of the concatenated tensor

#         return self.double_conv(x)