import argparse
import csv
import os
import os.path as osp
import datetime
import numpy as np
import torch
from icecream import ic
from tqdm import tqdm, trange
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResUNet
from evaluate import evaluate
from utils import set_seed, dice_score, BCEDiceLoss


ic.configureOutput(prefix="ic|", includeContext=True)


def train(epoch, net, data_loader, criterion, optimizer, args):
    train_loss_hist = []
    train_dice_hist = []

    net.train()

    for data in tqdm(
        data_loader,
        desc=f"Epoch {epoch+1}/{args.epochs}",
        dynamic_ncols=True,
        position=1,
        unit="imgs",
        unit_scale=args.batch_size,
        colour="yellow",
    ):
        img = data["image"].to(args.device)
        gt_mask = data["mask"].to(args.device)

        pred = net(img).squeeze(1)
        loss = criterion(pred, gt_mask.float())
        dice = dice_score(pred, gt_mask)

        train_loss_hist.append(loss.item())
        train_dice_hist.append(dice.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

    train_loss = float(np.mean(train_loss_hist))
    train_dice = float(np.mean(train_dice_hist))

    return train_loss, train_dice


def run(args):
    writer = SummaryWriter(log_dir=f"{args.out_dir}/logs")

    train_data = load_dataset(args.data_path, "train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    val_data = load_dataset(args.data_path, "valid")
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    match args.model:
        case "unet":
            model = UNet(3, 1).to(args.device)
        case "resunet":
            model = ResUNet(3, 1).to(args.device)
        case _:
            raise ValueError("Invalid model name")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    criterion = BCEDiceLoss()

    best_dice_score = 0

    epochs: dict[str, list[float]] = {
        "train_loss": [],
        "train_dice": [],
        "val_loss": [],
        "val_dice": [],
        "epoch": [],
    }

    for epoch in trange(
        args.epochs,
        desc="Epochs",
        dynamic_ncols=True,
        position=0,
        colour="green",
        leave=False,
    ):
        tqdm.write(f"Epoch {epoch+1}/{args.epochs} of training")
        train_loss, train_dice = train(
            epoch,
            model,
            train_loader,
            criterion,
            optimizer,
            args,
        )

        val_loss, val_dice = evaluate(
            model,
            val_loader,
            criterion,
            args,
        )

        tqdm.write(
            f"Epoch: {epoch+1}/{args.epochs}, Train Loss: {np.mean(train_loss):.4f}, Train Dice: {np.mean(train_dice):.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}"
        )

        torch.save(model.state_dict(), osp.join(args.out_dir, f"model_{epoch}.pth"))

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        writer.add_scalar("Dice/train", train_dice, epoch + 1)
        writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("Dice/val", val_dice, epoch + 1)

        epochs["epoch"].append(epoch + 1)
        epochs["train_loss"].append(train_loss)
        epochs["train_dice"].append(train_dice)
        epochs["val_loss"].append(val_loss)
        epochs["val_dice"].append(val_dice)

        if val_dice > best_dice_score:
            best_dice_score = val_dice
            tqdm.write(f"Saving best model with Dice Score: {best_dice_score:.4f}")
            torch.save(model.state_dict(), osp.join(args.out_dir, "best_model.pth"))

        with open(osp.join(args.out_dir, "epochs.csv"), "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(epochs.keys())
            csv_writer.writerows(zip(*epochs.values()))


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
        default=100,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-4,
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
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="model name",
        choices=["unet", "resunet"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to run the model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    out_dir = osp.join("log", args.output_dir)
    if osp.exists(out_dir):
        overwrite = input(
            f"Output directory {out_dir} already exists. Delete? [y/n]: ",
        )
        if overwrite.lower() != "y":
            exit(1)
        os.system(f"rm -rf {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    args.out_dir = out_dir
    delattr(args, "output_dir")

    set_seed(args.seed)

    ic(args)

    config_fname = osp.join(args.out_dir, "config.txt")
    with open(config_fname, "w") as f:
        f.write(str(args))

    # OxfordPetDataset.download(args.data_path)
    run(args)
