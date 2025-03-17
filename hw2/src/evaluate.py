import torch
import os.path as osp
from torch import nn

from utils import show_result, dice_score


def evaluate(net, dataloader, device, epoch, args):
    criterion = nn.BCELoss()
    with torch.no_grad():
        net.eval()
        val_loss = []
        val_dice = []
        for data in dataloader:
            img = data["image"].to(device)
            gt_mask = data["mask"].to(device)

            pred = net(img).squeeze(1)
            loss = criterion(pred, gt_mask.float())
            dice = dice_score(pred, gt_mask)

            val_loss.append(loss.item())
            val_dice.append(dice)

            show_result(
                img[0].permute(1, 2, 0).cpu().numpy().astype("uint8"),
                gt_mask[0].cpu().numpy().astype("uint8"),
                pred[0].cpu().numpy().astype("uint8"),
                osp.join(args.out_dir, f"results_{epoch}_{data['fname']}"),
            )

    avg_loss = sum(val_loss) / len(val_loss)
    avg_dice = sum(val_dice) / len(val_dice)
    return avg_loss, avg_dice
