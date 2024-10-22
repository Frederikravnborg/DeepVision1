from torch import nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncDec(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(size=16, mode='bilinear', align_corners=True)
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(size=32, mode='bilinear', align_corners=True)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(size=64, mode='bilinear', align_corners=True)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(size=128, mode='bilinear', align_corners=True)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = x.to(device)
        # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  

        return d3
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))
        skip0 = e0
        e0 = self.pool0(e0)

        e1 = F.relu(self.enc_conv1(e0))
        skip1 = e1
        e1 = self.pool1(e1)
        
        e2 = F.relu(self.enc_conv2(e1))
        skip2 = e2
        e2 = self.pool2(e2)

        e3 = F.relu(self.enc_conv3(e2))
        skip3 = e3
        e3 = self.pool3(e3)

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # Decoder
        d0 = self.upsample0(b)
        d0 = torch.cat([d0, skip3], 1)
        d0 = F.relu(self.dec_conv0(d0))
        
        d1 = self.upsample1(d0)
        d1 = torch.cat([d1, skip2], 1)
        d1 = F.relu(self.dec_conv1(d1))
        
        d2 = self.upsample2(d1)
        d2 = torch.cat([d2, skip1], 1)
        d2 = F.relu(self.dec_conv2(d2))
        
        d3 = self.upsample3(d2)
        d3 = torch.cat([d3, skip0], 1)
        d3 = self.dec_conv3(d3)
        
        return d3
    

class UNet2(nn.Module):
    def __init__(self):
        super(UNet2, self).__init__()

        # Encoder (downsampling using strided convolutions)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, stride=2, padding=1)  # stride=2 for downsampling
        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # stride=2 for downsampling
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # stride=2 for downsampling
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # stride=2 for downsampling

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder (upsampling using transpose convolutions)
        self.dec_tconv0 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))
        skip0 = e0

        e1 = F.relu(self.enc_conv1(e0))
        skip1 = e1
        
        e2 = F.relu(self.enc_conv2(e1))
        skip2 = e2

        e3 = F.relu(self.enc_conv3(e2))
        skip3 = e3

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # Decoder
        d0 = self.dec_tconv0(b)
        print(d0.shape, skip3.shape)
        d0 = torch.cat([d0, skip3], 1)
        d0 = F.relu(self.dec_conv0(d0))
        
        d1 = self.dec_tconv1(d0)
        d1 = torch.cat([d1, skip2], 1)
        d1 = F.relu(self.dec_conv1(d1))
        
        d2 = self.dec_tconv2(d1)
        d2 = torch.cat([d2, skip1], 1)
        d2 = F.relu(self.dec_conv2(d2))
        
        d3 = self.dec_tconv3(d2)
        d3 = torch.cat([d3, skip0], 1)
        d3 = self.dec_conv3(d3)
        
        return d3


class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # We use the same value for padding and dilation in each layer to maintain the spatial dimensions of the feature maps.
        # Specifically, by setting the padding equal to the dilation, we ensure that the output feature map has the same size as the input.
        # This helps in preserving the spatial resolution throughout the network.
        # The padding compensates for the increased receptive field caused by dilation, preventing any reduction in the feature map size.

        # encoder with dilated convolutions instead of downsampling
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1, dilation=1)  # dilation=1
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # dilation=2
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # dilation=4
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)  # dilation=8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder with dilated convolutions instead of upsampling
        self.dec_conv0 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)  # dilation=8
        self.dec_conv1 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # dilation=4
        self.dec_conv2 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # dilation=2
        self.dec_conv3 = nn.Conv2d(64, 1, 3, padding=1, dilation=1)  # dilation=1

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e1 = F.relu(self.enc_conv1(e0))
        e2 = F.relu(self.enc_conv2(e1))
        e3 = F.relu(self.enc_conv3(e2))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(b))
        d1 = F.relu(self.dec_conv1(d0))
        d2 = F.relu(self.dec_conv2(d1))
        d3 = self.dec_conv3(d2)

        return d3