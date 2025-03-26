from tqdm import tqdm
import torch

from utils import dice_score, save_figure


def evaluate(net, dataloader, criterion, args, position=1, save_results=False):
    val_loss = []
    val_dice = []

    net.eval()

    with torch.no_grad():
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

            if save_results:
                for i in range(img.size(0)):
                    save_figure(
                        gt_mask[i],
                        pred[i],
                        args.out_dir,
                        f"{data['image_name'][i]}.png",
                    )

    avg_loss = sum(val_loss) / len(val_loss)
    avg_dice = sum(val_dice) / len(val_dice)
    return avg_loss, avg_dice
