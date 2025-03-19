import numpy as np
import os
import torch
from PIL import Image
from icecream import ic
from torch import tensor
from torch import nn


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


class BCELoss:
    def __init__(self):
        self.bce = nn.BCELoss()

    def __call__(self, pred, target):
        return self.bce(pred, target.float())


class DiceLoss:
    def __init__(self):
        self.smooth = 1e-6

    def __call__(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


class BCEDiceLoss:
    def __init__(self):
        self.bce = BCELoss()
        self.dice = DiceLoss()

    def __call__(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)
