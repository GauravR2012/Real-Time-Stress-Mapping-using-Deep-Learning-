# models/encoder.py

import torch
import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self,in_channels,out_channels):

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.block(x)


class Encoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = ConvBlock(3,64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(64,128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(128,256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(256,512)
        self.pool4 = nn.MaxPool2d(2)

    def forward(self,x):

        x1 = self.conv1(x)
        p1 = self.pool1(x1)

        x2 = self.conv2(p1)
        p2 = self.pool2(x2)

        x3 = self.conv3(p2)
        p3 = self.pool3(x3)

        x4 = self.conv4(p3)

        return x1,x2,x3,x4
