import torch
import torch.nn as nn

class WhatsCookingNet(nn.Module):
    
    def __init__(self, in_features=6714):
        super(WhatsCookingNet, self).__init__()
        self.blk = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 20)
        )

    def forward(self, x):
        return self.blk(x)