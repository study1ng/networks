# Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., & Bengio, Y. (2013, May). Maxout networks. In International conference on machine learning(pp. 1319-1327). PMLR.
# maxout network
import torch
from torch import nn


class Maxout(nn.Module):
    # Maxout activation function
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(dim=-1)[0]
