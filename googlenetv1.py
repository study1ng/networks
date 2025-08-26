# Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).
import torch
import torch.nn as nn

import lightning


class InceptionModule(nn.Module):
    # Inception Module with dimension reduction
    def __init__(self, in_channels, x1, x3, x3red, x5, x5red, poolproj):
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, x1, kernel_size=1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, x3red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(x3red, x3, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, x5red, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(x5red, x5, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, poolproj, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.cat((
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ), dim=1)


class GoogleNet(lightning.LightningModule):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(192, 64, 128, 96, 32, 16, 32),
            InceptionModule(256, 128, 192, 128, 96, 32, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.inception_with_aux1 = InceptionModule(
            480, 192, 208, 96, 48, 16, 64)
        self.layers2 = nn.Sequential(
            InceptionModule(512, 160, 224, 112, 64, 24, 64),
            InceptionModule(512, 112, 288, 144, 64, 32, 64),
        )
        self.inception_with_aux2 = InceptionModule(
            528, 256, 320, 160, 128, 32, 128)
        self.layers3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionModule(832, 256, 320, 160, 128, 32, 128),
            InceptionModule(1024, 384, 384, 192, 128, 48, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes),
        )
        self.aux1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.aux2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Conv2d(528, 128, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes),
        )
        self.loss = nn.CrossEntropyLoss()
        self.auxloss1 = nn.CrossEntropyLoss()
        self.auxloss2 = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.layers1(x)
        x = self.inception_with_aux1(x)
        o1 = self.aux1(x)
        x = self.layers2(x)
        x = self.inception_with_aux2(x)
        o2 = self.aux2(x)
        x = self.layers3(x)
        if self.training:
            return x, o1, o2
        return x

    def training_step(self, batch, _):
        x, t = batch
        y, o1, o2 = self(x)
        loss = self.loss(y, t)
        auxloss1 = self.auxloss1(o1, t)
        auxloss2 = self.auxloss2(o2, t)
        loss += 0.3 * auxloss1 + 0.3 * auxloss2
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9)

        def lr_lambda(epoch): return 0.96 ** (epoch // 8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda)
        return [optimizer], [scheduler]
