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
