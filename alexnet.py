# Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25.
# Implemented AlexNet model
# In AlexNet, Dropout layer's output is multiplied with 0.5, but in torch, nn.Dropout is inverted so we don't need such technique.
import torch
from torch import nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.pregpu = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # AlexNet separates the first conv layer's output to process on different gpu.
        # Though pytorch implicitly does this, to reproduce the implementation based the original paper, we explicitly separate them.
        # Actually, this code does not process parallel. Just for imitation.
        self.gpu1 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.gpu2 = nn.Sequential(
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.postgpu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.pregpu(x)
        x1 = self.gpu1(x[:, :48, ...])
        x2 = self.gpu2(x[:, 48:, ...])
        x = torch.cat([x1, x2], dim=1)
        x = self.postgpu(x)
        return x

    def _initialize_weights(self):
        # By default, torch initialize weights with Kaiming initialization. To reproduce the original paper, we use normal initialize.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


net = AlexNet()
print(net)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
