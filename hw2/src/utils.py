import torch
from torch import tensor
from icecream import ic
import numpy as np


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    """
    Compute Dice Score for binary segmentation.
    Args:
      pred: Tensor of shape (N, 2, H, W), raw logits.
      target: Tensor of shape (N, H, W), containing {0,1}.
      threshold: Threshold for converting logits to binary mask.
      eps: Small value to avoid division by zero.
    Returns:
      Dice coefficient (scalar).
    """
    # Apply softmax and take the foreground channel (index 1)
    pred = torch.softmax(pred, dim=1)[:, 1, :, :]  # Shape: (N, H, W)

    # Convert to binary mask
    pred_bin = (pred > threshold).float()
    target = target.float()

    intersection = (pred_bin * target).sum(dim=(1, 2))  # Per image
    union = pred_bin.sum(dim=(1, 2)) + target.sum(dim=(1, 2))  # Per image

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()  # Average over batch


def dice_loss(pred_mask, gt_mask):
    return 1 - dice_score(pred_mask, gt_mask)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
