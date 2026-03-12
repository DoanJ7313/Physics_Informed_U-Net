import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1)
        
        # self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
        # self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels, affine=True)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        # self.norm2 = nn.BatchNorm2d(out_channels)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        identity = self.conv(x)
        out = self.pool(identity)
        return identity, out

class UpSample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = UpConvBlock(in_channels + skip_channels, out_channels)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(x1, size=x2.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=4, base=64):
        super().__init__()
        self.down1 = DownSample(in_channels, base)
        self.down2 = DownSample(base, base * 2)
        self.down3 = DownSample(base * 2, base * 4)
        self.down4 = DownSample(base * 4, base * 8)

        self.bottle_neck = ConvBlock(base * 8, base * 16)

        self.up1 = UpSample(base * 16, base * 8, base * 8)
        self.up2 = UpSample(base * 8, base * 4, base * 4)
        self.up3 = UpSample(base * 4, base * 2, base * 2)
        self.up4 = UpSample(base * 2, base, base)

        self.out = nn.Conv2d(base, 2, kernel_size=1)
    
    def forward(self, x):

        x1, x = self.down1(x)
        x2, x = self.down2(x)
        x3, x = self.down3(x)
        x4, x = self.down4(x)

        x = self.bottle_neck(x)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x = self.out(x)
        return x
