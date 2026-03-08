# models/force_decoder.py

import torch
import torch.nn as nn


class ForceDecoder(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc = nn.Sequential(

            nn.Linear(input_dim,128),
            nn.Sigmoid(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,32),
            nn.ReLU(),

            nn.Linear(32,8),
            nn.ReLU(),

            nn.Linear(8,2)
        )

    def forward(self,x):

        return self.fc(x)
