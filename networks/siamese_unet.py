import os
import torch
import math
from torch import nn
import torch.nn.functional as F
from networks.custom_modules.basic_modules import *


'''
================================================================
Total params: 43,782,132
Trainable params: 43,782,132
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 149.84
Params size (MB): 167.02
Estimated Total Size (MB): 316.86
----------------------------------------------------------------
'''


class Baseline(nn.Module):
    def __init__(self, img_ch=1, num_classes=6, depth=3):
        super(Baseline, self).__init__()
        chs = [36, 72, 144, 288, 360]
        chs2 = [36, 72, 144, 288, 360]
        self.pool = nn.MaxPool2d(2, 2)
        # first path
        self.p1_enc1 = EncoderBlock(img_ch, chs[0], depth=depth)
        self.p1_enc2 = EncoderBlock(chs[0], chs[1], depth=depth)
        self.p1_enc3 = EncoderBlock(chs[1], chs[2], depth=depth)
        self.p1_enc4 = EncoderBlock(chs[2], chs[3], depth=depth)
        self.p1_cen = EncoderBlock(chs[3], chs[4], depth=depth)

        self.p1_dec4 = DecoderBlock(chs[4] * 3 + chs2[4] * 3, chs[3])
        self.p1_decconv4 = EncoderBlock(chs[3] * 4 + chs2[3] * 3, chs[3])
        self.p1_dec3 = DecoderBlock(chs[3], chs[2])
        self.p1_decconv3 = EncoderBlock(chs[2] * 4 + chs2[2] * 3, chs[2])

        self.p1_dec2 = DecoderBlock(chs[2], chs[1])
        self.p1_decconv2 = EncoderBlock(chs[1] * 4 + chs2[1] * 3, chs[1])

        self.p1_dec1 = DecoderBlock(chs[1], chs[0])
        self.p1_decconv1 = EncoderBlock(chs[0] * 4 + chs2[0] * 3, chs[0])

        self.p1_conv_1x1 = nn.Conv2d(chs[0], num_classes, kernel_size=1, stride=1, padding=0)

        # second path
        self.p2_enc1 = EncoderBlock(img_ch, chs2[0], depth=depth)
        self.p2_enc2 = EncoderBlock(chs2[0], chs2[1], depth=depth)
        self.p2_enc3 = EncoderBlock(chs2[1], chs2[2], depth=depth)
        self.p2_enc4 = EncoderBlock(chs2[2], chs2[3], depth=depth)
        self.p2_cen = EncoderBlock(chs2[3], chs2[4], depth=depth)

        self.p2_dec4 = DecoderBlock(chs2[4] * 3, chs2[3])
        self.p2_decconv4 = EncoderBlock(chs2[3] * 4, chs2[3])
        self.p2_dec3 = DecoderBlock(chs2[3], chs2[2])
        self.p2_decconv3 = EncoderBlock(chs2[2] * 4, chs2[2])

        self.p2_dec2 = DecoderBlock(chs2[2], chs2[1])
        self.p2_decconv2 = EncoderBlock(chs2[1] * 4, chs2[1])

        self.p2_dec1 = DecoderBlock(chs2[1], chs2[0])
        self.p2_decconv1 = EncoderBlock(chs2[0] * 4, chs2[0])

        self.p2_conv_1x1 = nn.Conv2d(chs2[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2, x3):
        """
        first path
        """
        # p1 encoder
        p1_x1 = self.p1_enc1(x1)
        p1_x2 = self.pool(p1_x1)
        p1_x2 = self.p1_enc2(p1_x2)
        p1_x3 = self.pool(p1_x2)
        p1_x3 = self.p1_enc3(p1_x3)
        p1_x4 = self.pool(p1_x3)
        p1_x4 = self.p1_enc4(p1_x4)
        p1_center = self.pool(p1_x4)
        p1_center = self.p1_cen(p1_center)

        # p2 encoder
        p2_x1 = self.p1_enc1(x2)
        p2_x2 = self.pool(p2_x1)
        p2_x2 = self.p1_enc2(p2_x2)
        p2_x3 = self.pool(p2_x2)
        p2_x3 = self.p1_enc3(p2_x3)
        p2_x4 = self.pool(p2_x3)
        p2_x4 = self.p1_enc4(p2_x4)
        p2_center = self.pool(p2_x4)
        p2_center = self.p1_cen(p2_center)

        # p3 encoder
        p3_x1 = self.p1_enc1(x3)
        p3_x2 = self.pool(p3_x1)
        p3_x2 = self.p1_enc2(p3_x2)
        p3_x3 = self.pool(p3_x2)
        p3_x3 = self.p1_enc3(p3_x3)
        p3_x4 = self.pool(p3_x3)
        p3_x4 = self.p1_enc4(p3_x4)
        p3_center = self.pool(p3_x4)
        p3_center = self.p1_cen(p3_center)

        fuse_center = torch.cat([p1_center, p2_center, p3_center], dim=1)
        fuse4 = torch.cat([p1_x4, p2_x4, p3_x4], dim=1)
        fuse3 = torch.cat([p1_x3, p2_x3, p3_x3], dim=1)
        fuse2 = torch.cat([p1_x2, p2_x2, p3_x2], dim=1)
        fuse1 = torch.cat([p1_x1, p2_x1, p3_x1], dim=1)

        """
        second path
        """
        # p1 encoder
        p2_p1_x1 = self.p2_enc1(x1)
        p2_p1_x2 = self.pool(p2_p1_x1)
        p2_p1_x2 = self.p2_enc2(p2_p1_x2)
        p2_p1_x3 = self.pool(p2_p1_x2)
        p2_p1_x3 = self.p2_enc3(p2_p1_x3)
        p2_p1_x4 = self.pool(p2_p1_x3)
        p2_p1_x4 = self.p2_enc4(p2_p1_x4)
        p2_p1_center = self.pool(p2_p1_x4)
        p2_p1_center = self.p2_cen(p2_p1_center)

        # p2 encoder
        p2_p2_x1 = self.p2_enc1(x2)
        p2_p2_x2 = self.pool(p2_p2_x1)
        p2_p2_x2 = self.p2_enc2(p2_p2_x2)
        p2_p2_x3 = self.pool(p2_p2_x2)
        p2_p2_x3 = self.p2_enc3(p2_p2_x3)
        p2_p2_x4 = self.pool(p2_p2_x3)
        p2_p2_x4 = self.p2_enc4(p2_p2_x4)
        p2_p2_center = self.pool(p2_p2_x4)
        p2_p2_center = self.p2_cen(p2_p2_center)

        # p3 encoder
        p2_p3_x1 = self.p2_enc1(x3)
        p2_p3_x2 = self.pool(p2_p3_x1)
        p2_p3_x2 = self.p2_enc2(p2_p3_x2)
        p2_p3_x3 = self.pool(p2_p3_x2)
        p2_p3_x3 = self.p2_enc3(p2_p3_x3)
        p2_p3_x4 = self.pool(p2_p3_x3)
        p2_p3_x4 = self.p2_enc4(p2_p3_x4)
        p2_p3_center = self.pool(p2_p3_x4)
        p2_p3_center = self.p2_cen(p2_p3_center)

        p2_fuse_center = torch.cat([p2_p1_center, p2_p2_center, p2_p3_center], dim=1)
        p2_fuse4 = torch.cat([p2_p1_x4, p2_p2_x4, p2_p3_x4], dim=1)
        p2_fuse3 = torch.cat([p2_p1_x3, p2_p2_x3, p2_p3_x3], dim=1)
        p2_fuse2 = torch.cat([p2_p1_x2, p2_p2_x2, p2_p3_x2], dim=1)
        p2_fuse1 = torch.cat([p2_p1_x1, p2_p2_x1, p2_p3_x1], dim=1)

        p2_d4 = self.p2_dec4(p2_fuse_center)
        p2_d4 = torch.cat((p2_fuse4, p2_d4), dim=1)
        p2_d4 = self.p2_decconv4(p2_d4)

        p2_d3 = self.p2_dec3(p2_d4)
        p2_d3 = torch.cat((p2_fuse3, p2_d3), dim=1)
        p2_d3 = self.p2_decconv3(p2_d3)

        p2_d2 = self.p2_dec2(p2_d3)
        p2_d2 = torch.cat((p2_fuse2, p2_d2), dim=1)
        p2_d2 = self.p2_decconv2(p2_d2)

        p2_d1 = self.p2_dec1(p2_d2)
        p2_d1 = torch.cat((p2_fuse1, p2_d1), dim=1)
        p2_d1 = self.p2_decconv1(p2_d1)

        p2_out = self.p2_conv_1x1(p2_d1)


        """
        first path decoder
        """
        d4 = self.p1_dec4(torch.cat([fuse_center, p2_fuse_center], dim=1))
        d4 = torch.cat((fuse4, d4, p2_fuse4), dim=1)
        d4 = self.p1_decconv4(d4)

        d3 = self.p1_dec3(d4)
        d3 = torch.cat((fuse3, d3, p2_fuse3), dim=1)
        d3 = self.p1_decconv3(d3)

        d2 = self.p1_dec2(d3)
        d2 = torch.cat((fuse2, d2, p2_fuse2), dim=1)
        d2 = self.p1_decconv2(d2)

        d1 = self.p1_dec1(d2)
        d1 = torch.cat((fuse1, d1, p2_fuse1), dim=1)
        d1 = self.p1_decconv1(d1)

        p1_out = self.p1_conv_1x1(d1)

        return p1_out, p2_out

if __name__ == '__main__':
    from torchsummary import summary

    x1 = torch.randn([2, 1, 256, 256]).cuda()
    x2 = torch.randn([2, 1, 256, 256]).cuda()
    x3 = torch.randn([2, 1, 256, 256]).cuda()
    net = Baseline(num_classes=6, depth=2).cuda()
    # summary(net, input_size=[(1, 64, 64), (1, 64, 64), (1, 64, 64)])
    pred = net(x1, x2, x3)
    print(pred[0].shape)
