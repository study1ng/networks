# Lin, M., Chen, Q., & Yan, S. (2013). Network in network. arXiv preprint arXiv:1312.4400.
# Network in Network, Implemented MLPConv layer
import torch
from torch import nn


class MLPConv(nn.Module):
    def __init__(self, f_in, f_out, mlp_size):
        super(MLPConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(f_in, f_out, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        for _ in range(mlp_size):
            self.layers.append(nn.Conv2d(f_out, f_out, 1, 1, 0))
            self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class NIN(nn.Module):
    # Network in Network
    # I didn't implemented the weight initialize
    def __init__(self, num_classes):
        super(NIN, self).__init__()
        self.layers = nn.Sequential(  # I could not find the supplementary materials in the paper so the parameters is my setting
            MLPConv(3, 192, 2),
            nn.Dropout(0.5),
            nn.MaxPool2d(3, 2, 1),
            MLPConv(192, 192, 2),
            nn.Dropout(0.5),
            nn.MaxPool2d(3, 2, 1),
            MLPConv(192, num_classes, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.layers(x)
