from icecream import ic
import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding="valid"),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding="valid"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                DoubleConv(3, 64),  # 0
                DoubleConv(64, 128),  # 1
                DoubleConv(128, 256),  # 2
                DoubleConv(256, 512),  # 3
                DoubleConv(512, 1024),  # 4
                nn.ConvTranspose2d(1024, 512, 2, stride=2),  # 5
                DoubleConv(1024, 512),  # 6
                nn.ConvTranspose2d(512, 256, 2, stride=2),  # 7
                DoubleConv(512, 256),  # 8
                nn.ConvTranspose2d(256, 128, 2, stride=2),  # 9
                DoubleConv(256, 128),  # 10
                nn.ConvTranspose2d(128, 64, 2, stride=2),  # 11
                DoubleConv(128, 64),  # 12
                nn.Conv2d(64, 2, 1),  # 13
            ]
        )

    def forward(self, x):
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](nn.MaxPool2d(2)(x1))
        x3 = self.encoder[2](nn.MaxPool2d(2)(x2))
        x4 = self.encoder[3](nn.MaxPool2d(2)(x3))
        x5 = self.encoder[4](nn.MaxPool2d(2)(x4))
        x6 = torch.cat((x4[:, :, 4:60, 4:60], self.encoder[5](x5)), dim=1)
        x7 = self.encoder[6](x6)
        x8 = torch.cat((x3[:, :, 16:120, 16:120], self.encoder[7](x7)), dim=1)
        x9 = self.encoder[8](x8)
        x10 = torch.cat((x2[:, :, 40:240, 40:240], self.encoder[9](x9)), dim=1)
        x11 = self.encoder[10](x10)
        x12 = torch.cat((x1[:, :, 88:480, 88:480], self.encoder[11](x11)), dim=1)
        x13 = self.encoder[12](x12)
        ret = self.encoder[13](x13)
        return ret
