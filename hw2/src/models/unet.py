from icecream import ic
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="same", bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="same", bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.encoder = nn.ModuleList(
            [
                DoubleConv(in_channels, 64),  # 0
                DoubleConv(64, 128),  # 1
                DoubleConv(128, 256),  # 2
                DoubleConv(256, 512),  # 3
                DoubleConv(512, 1024),  # 4
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 0
                DoubleConv(1024, 512),  # 1
                nn.ConvTranspose2d(512, 256, 2, stride=2),  # 2
                DoubleConv(512, 256),  # 3
                nn.ConvTranspose2d(256, 128, 2, stride=2),  # 4
                DoubleConv(256, 128),  # 5
                nn.ConvTranspose2d(128, 64, 2, stride=2),  # 6
                DoubleConv(128, 64),  # 7
                nn.Sequential(
                    nn.Conv2d(64, out_channels, 1),
                    nn.Sigmoid(),
                ),  # 8
            ]
        )

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](self.pool(x1))
        x3 = self.encoder[2](self.pool(x2))
        x4 = self.encoder[3](self.pool(x3))
        x5 = self.encoder[4](self.pool(x4))

        x6 = torch.cat((x4, self.decoder[0](x5)), dim=1)
        x7 = self.decoder[1](x6)
        x8 = torch.cat((x3, self.decoder[2](x7)), dim=1)
        x9 = self.decoder[3](x8)
        x10 = torch.cat((x2, self.decoder[4](x9)), dim=1)
        x11 = self.decoder[5](x10)
        x12 = torch.cat((x1, self.decoder[6](x11)), dim=1)
        x13 = self.decoder[7](x12)

        ret = self.decoder[8](x13)

        return ret
