# models/stress_decoder.py

import torch
import torch.nn as nn


class UpBlock(nn.Module):

    def __init__(self,in_channels,out_channels):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels,out_channels,2,stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,skip):

        x = self.up(x)

        x = torch.cat([x,skip],dim=1)

        x = self.conv(x)

        return x


class StressDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.up1 = UpBlock(512,256)
        self.up2 = UpBlock(256,128)
        self.up3 = UpBlock(128,64)

        self.final = nn.Conv2d(64,1,1)

    def forward(self,x1,x2,x3,x4):

        d1 = self.up1(x4,x3)

        d2 = self.up2(d1,x2)

        d3 = self.up3(d2,x1)

        out = self.final(d3)

        return out
