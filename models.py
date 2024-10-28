from torch import nn
import torch.nn.functional as F
import torch

device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")

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
    
    def name(self):
        return 'EncDec'

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
        
        # d3 = torch.sigmoid(d3)

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
    
    def name(self):
        return 'UNet'

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
        self.enc_conv0 = nn.Conv2d(3, 64, 3, stride=1, padding=1)    # Keep stride=1 here
        self.enc_conv1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)   # Downsampling starts here
        self.enc_conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # Decoder (upsampling using transpose convolutions)
        self.dec_tconv0 = nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1, output_padding=0)  # Keep stride=1
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv2 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_tconv3 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def name(self):
        return 'UNet2'

    def forward(self, x):
        # Encoder
        e0 = F.relu(self.enc_conv0(x))     # Shape: [batch, 64, H, W]
        skip0 = e0                         # Skip connection at full resolution

        e1 = F.relu(self.enc_conv1(e0))    # Shape: [batch, 64, H/2, W/2]
        skip1 = e1

        e2 = F.relu(self.enc_conv2(e1))    # Shape: [batch, 64, H/4, W/4]
        skip2 = e2

        e3 = F.relu(self.enc_conv3(e2))    # Shape: [batch, 64, H/8, W/8]
        skip3 = e3

        # Bottleneck
        b = F.relu(self.bottleneck_conv(e3))  # Shape: [batch, 64, H/8, W/8]

        # Decoder
        d0 = self.dec_tconv0(b)            # Shape: [batch, 64, H/8, W/8]
        d0 = torch.cat([d0, skip3], dim=1)  # Concatenate with skip3
        d0 = F.relu(self.dec_conv0(d0))

        d1 = self.dec_tconv1(d0)           # Shape: [batch, 64, H/4, W/4]
        d1 = torch.cat([d1, skip2], dim=1)
        d1 = F.relu(self.dec_conv1(d1))

        d2 = self.dec_tconv2(d1)           # Shape: [batch, 64, H/2, W/2]
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = F.relu(self.dec_conv2(d2))

        d3 = self.dec_tconv3(d2)           # Shape: [batch, 64, H, W]
        d3 = torch.cat([d3, skip0], dim=1)
        d3 = self.dec_conv3(d3)

        return d3


class UNet2batch(nn.Module):
    def __init__(self, n_classes=1):
        super(UNet2batch, self).__init__()

        # Encoder (Downsampling with strided convs)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc_bn0 = nn.BatchNorm2d(64)
        
        self.enc_conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(128)
        
        self.enc_conv2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(256)

        self.enc_conv3 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.enc_bn3 = nn.BatchNorm2d(512)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(512, 1024, 3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(1024)

        # Decoder (Upsampling with transpose convolutions)
        self.dec_tconv0 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv0 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dec_bn0 = nn.BatchNorm2d(512)

        self.dec_tconv1 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(256)

        self.dec_tconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(128)

        self.dec_tconv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(64)

        # Final layer (output)
        self.final_conv = nn.Conv2d(64, n_classes, 1)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Encoder
        e0 = F.leaky_relu(self.enc_bn0(self.enc_conv0(x)))    # Shape: [batch, 64, H, W]
        skip0 = e0

        e1 = F.leaky_relu(self.enc_bn1(self.enc_conv1(e0)))   # Shape: [batch, 128, H/2, W/2]
        skip1 = e1

        e2 = F.leaky_relu(self.enc_bn2(self.enc_conv2(e1)))   # Shape: [batch, 256, H/4, W/4]
        skip2 = e2

        e3 = F.leaky_relu(self.enc_bn3(self.enc_conv3(e2)))   # Shape: [batch, 512, H/8, W/8]
        skip3 = e3

        # Bottleneck
        b = F.leaky_relu(self.bottleneck_bn(self.bottleneck_conv(e3)))  # Shape: [batch, 1024, H/8, W/8]

        # Decoder
        d0 = F.leaky_relu(self.dec_tconv0(b))                   # Shape: [batch, 512, H/4, W/4]
        d0 = torch.cat([d0, skip3], dim=1)                      # Concatenate skip connection
        d0 = F.leaky_relu(self.dec_bn0(self.dec_conv0(d0)))

        d1 = F.leaky_relu(self.dec_tconv1(d0))                  # Shape: [batch, 256, H/2, W/2]
        d1 = torch.cat([d1, skip2], dim=1)
        d1 = F.leaky_relu(self.dec_bn1(self.dec_conv1(d1)))

        d2 = F.leaky_relu(self.dec_tconv2(d1))                  # Shape: [batch, 128, H, W]
        d2 = torch.cat([d2, skip1], dim=1)
        d2 = F.leaky_relu(self.dec_bn2(self.dec_conv2(d2)))

        d3 = F.leaky_relu(self.dec_tconv3(d2))                  # Shape: [batch, 64, H, W]
        d3 = torch.cat([d3, skip0], dim=1)
        d3 = F.leaky_relu(self.dec_bn3(self.dec_conv3(d3)))

        # Final output layer (sigmoid for binary, softmax for multi-class)
        output = self.final_conv(d3)
        output = torch.sigmoid(output) if self.final_conv.out_channels == 1 else torch.softmax(output, dim=1)

        return output

    def name(self):
        return 'UNet2batch'


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

    def name(self):
        return 'DilatedNet'

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
