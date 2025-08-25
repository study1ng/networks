import torch
from torch import nn


class MLPConv(nn.Module):
    def __init__(self, f_in, f_out):
        super(MLPConv, self).__init__()
        self.layers = nn.Sequential([
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
            nn.Linear(),
            
        ])