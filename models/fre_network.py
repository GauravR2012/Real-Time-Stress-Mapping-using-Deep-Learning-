# models/fre_network.py

import torch
import torch.nn as nn

from .encoder import Encoder
from .stress_decoder import StressDecoder
from .force_decoder import ForceDecoder


class FREModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = Encoder()

        self.stress_decoder = StressDecoder()

        self.force_decoder = ForceDecoder(512*14*14)


    def forward(self,x):

        x1,x2,x3,x4 = self.encoder(x)

        stress = self.stress_decoder(x1,x2,x3,x4)

        flattened = x4.view(x4.size(0),-1)

        force = self.force_decoder(flattened)

        return stress,force
