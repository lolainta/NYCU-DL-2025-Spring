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


def show_result(img, gt_mask, pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(img).save(out_dir + "/input.png")
    Image.fromarray((gt_mask * 255).astype(np.uint8)).save(out_dir + "/gt_mask.png")
    Image.fromarray((pred * 255).astype(np.uint8)).save(out_dir + "/pred_mask.png")


def dice_score(pred, target):
    smooth = 1e-6
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
