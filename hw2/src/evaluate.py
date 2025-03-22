from tqdm import tqdm
import torch

from utils import dice_score


def evaluate(net, dataloader, criterion, args, position=1):
    with torch.no_grad():
        net.eval()
        val_loss = []
        val_dice = []
        for data in tqdm(
            dataloader,
            desc="Evaluate",
            dynamic_ncols=True,
            position=position,
            unit="imgs",
            unit_scale=args.batch_size,
            colour="yellow",
        ):
            img = data["image"].to(args.device)
            gt_mask = data["mask"].to(args.device)

            pred = net(img).squeeze(1)
            loss = criterion(pred, gt_mask.float())
            dice = dice_score(pred, gt_mask)

            val_loss.append(loss.item())
            val_dice.append(dice.item())

    avg_loss = sum(val_loss) / len(val_loss)
    avg_dice = sum(val_dice) / len(val_dice)
    return avg_loss, avg_dice
