import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import initialize_weights


activation = nn.ReLU


class EncoderBlock(nn.Module):
    def __init__(self, ch_in, ch_out, depth=2, use_res=False):
        super(EncoderBlock, self).__init__()

        self.use_res = use_res

        self.conv = [nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            activation(),
        )]

        for i in range(1, depth):
            self.conv.append(nn.Sequential(nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                                           nn.BatchNorm2d(ch_out),
                                           nn.Sequential() if use_res and i == depth-1 else activation()
                                           ))
        self.conv = nn.Sequential(*self.conv)
        if use_res:
            self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        if self.use_res:
            residual = self.conv1x1(x)

        x = self.conv(x)

        if self.use_res:
            x += residual
            x = F.relu(x)

        return x


class DecoderBlock(nn.Module):
    """
    Interpolate
    """

    def __init__(self, ch_in, ch_out, use_deconv=False):
        super(DecoderBlock, self).__init__()
        if use_deconv:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch_out),
                activation()
            )

    def forward(self, x):
        return self.up(x)


class ResBlockV1(nn.Module):
    """
    Post-activation
    """

    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(ResBlockV1, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)

        return out


class ResBlockV2(nn.Module):
    """
    Post-activation
    """

    def __init__(self, ch_in, ch_out, kernel_size=3):
        super(ResBlockV2, self).__init__()

        self.bn1 = nn.BatchNorm2d(ch_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size, 1, kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size, 1, kernel_size // 2, bias=False)

        self.conv1x1 = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        residual = self.conv1x1(x)

        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))

        out += residual

        return out


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):

        return self.conv(x)



