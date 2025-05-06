import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel


class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, dim=512):
        super().__init__()
        channel = dim // 4
        self.UNet = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[
                channel,
                channel,
                channel * 2,
                channel * 2,
            ],  # type: ignore
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
            ],  # type: ignore
            up_block_types=[
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],  # type: ignore
            class_embed_type="identity",
        )
        self.label_embedding = nn.Linear(num_classes, dim)

    def forward(self, x, t, label):
        embedding_label = self.label_embedding(label)
        return self.UNet(x, t, embedding_label).sample
