import argparse
import os
import os.path as osp
import datetime
import numpy as np
import torch
from icecream import ic
from tqdm import tqdm, trange
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from oxford_pet import load_dataset
from models.unet import UNet
from evaluate import evaluate
from utils import set_seed, dice_score


ic.configureOutput(prefix="ic|", includeContext=True)


def train(args):
    ic(args)
    writer = SummaryWriter(log_dir=f"{args.out_dir}/logs")
    train_data = load_dataset(args.data_path, "train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = load_dataset(args.data_path, "valid")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = UNet(3, 1).to(args.device)
    # ic(model)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    criterion = nn.BCELoss()

    best_dice_score = 0

    cnt = 0
    for epoch in trange(args.epochs, desc="Epochs", position=0):
        model.train()

        train_loss_hist = []
        train_dice_hist = []

        for data in tqdm(train_loader, desc="Batches", position=1):
            img = data["image"].to(args.device)
            gt_mask = data["mask"].to(args.device)

            pred = model(img).squeeze(1)
            loss = criterion(pred, gt_mask.float())
            dice = dice_score(pred, gt_mask)
            # tqdm.write(f"Loss: {loss.item():.4f}, Dice: {dice:.4f}")

            writer.add_scalar("Loss/train", loss.item(), cnt)
            writer.add_scalar("Dice/train", dice.item(), cnt)

            train_loss_hist.append(loss.item())
            train_dice_hist.append(dice.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            cnt += 1

        train_loss = np.mean(train_loss_hist)
        train_dice = np.mean(train_dice_hist)

        val_loss, val_dice = evaluate(model, val_loader, args)

        tqdm.write(
            f"Epoch: {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_loss):.4f}, Train Dice: {np.mean(train_dice):.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
        )

        torch.save(model.state_dict(), osp.join(args.out_dir, f"model_{epoch}.pth"))

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Dice/val", val_dice, epoch)

        if val_dice > best_dice_score:
            best_dice_score = val_dice
            tqdm.write(f"Saving best model with Dice Score: {best_dice_score:.4f}")
            torch.save(model.state_dict(), osp.join(args.out_dir, "best_model.pth"))

    torch.save(model.state_dict(), osp.join(args.out_dir, "model.pth"))


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/oxford-iiit-pet/link",
        help="path of the input data",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=30,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-3,
        help="learning rate",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="output directory",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="random seed",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    out_dir = osp.join("log", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir = out_dir
    args.seed = set_seed(args.seed)

    delattr(args, "output_dir")

    ic(args)

    config_fname = osp.join(args.out_dir, "config.txt")
    with open(config_fname, "w") as f:
        f.write(str(args))

    # OxfordPetDataset.download(args.data_path)
    train(args)
