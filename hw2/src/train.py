import argparse
import torch
from icecream import ic
import numpy as np
from tqdm import tqdm, trange
from PIL import Image
from torch.utils.data import DataLoader
import os
import os.path as osp
import datetime

from oxford_pet import OxfordPetDataset, SimpleOxfordPetDataset
from models.unet import UNet
from utils import dice_score, set_seed

ic.configureOutput(prefix="ic|", includeContext=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trans(image, mask, trimap):
    ic.disable()
    ic(image.shape, mask.shape, trimap.shape)
    ret_image = np.array(
        Image.fromarray(image).resize((572, 572)), dtype=np.float32
    ).reshape((3, 572, 572))
    ret_mask = np.array(
        Image.fromarray(mask).resize((388, 388)), dtype=np.int64
    ).reshape((388, 388))
    ret_trimap = np.array(
        Image.fromarray(trimap).resize((388, 388)), dtype=np.float32
    ).reshape((1, 388, 388))
    ic(ret_image.shape, ret_mask.shape, ret_trimap.shape)
    ic.enable()
    return dict(image=ret_image, mask=ret_mask, trimap=ret_trimap)


def train(args):
    ic(args)

    config_fname = osp.join(args.out_dir, "config.txt")
    with open(config_fname, "w") as f:
        f.write(str(args))

    train_data = OxfordPetDataset(
        root=args.data_path,
        mode="train",
        transform=trans,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    model = UNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()  # Logits expected, applies softmax

    for epoch in trange(args.epochs, desc="Epochs", position=0):
        for data in tqdm(train_loader, desc="Batches", position=1):
            img = data["image"].to(device)
            gt_mask = data["mask"].to(device)

            pred = model(img)
            # pred_mask = pixel-wise softmax
            # pred_mask = torch.softmax(pred, dim=1)
            # pred_mask = torch.argmax(pred_mask, dim=1)

            # loss = loss_fn(pred_mask, gt_mask)
            loss = criterion(pred, gt_mask)
            score = dice_score(pred, gt_mask)
            tqdm.write(f"Loss: {loss.item()}, Dice Score: {score.item()}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), osp.join(args.out_dir, f"model_{epoch}.pth"))

    torch.save(model.state_dict(), osp.join(args.out_dir, "model.pth"))


def test(args):
    ic(args)
    test_data = OxfordPetDataset(
        root=args.data_path,
        mode="test",
        transform=trans,
    )

    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = UNet().to(device)
    model.load_state_dict(torch.load(osp.join(args.out_dir, "model.pth")))

    scores = []
    for data in tqdm(test_loader, desc="Testing", position=0):
        pred = model(data["image"].to(device))
        score = dice_score(pred, data["mask"].to(device))
        scores.append(score.item())

    scores = np.array(scores)
    print(f"Average Dice Score: {np.mean(scores)}")


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
        default=5,
        help="number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="output directory",
    )
    parser.add_argument("--seed", "-s", type=int, default=42, help="random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    out_dir = osp.join("log", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    args.device = device
    args.out_dir = out_dir
    args.seed = set_seed(args.seed)

    args.output_dir = None

    ic(args)

    # OxfordPetDataset.download(args.data_path)
    #
    train(args)
    test(args)
