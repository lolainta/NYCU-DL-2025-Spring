import argparse
import datetime
import os
import os.path as osp
from icecream import ic
import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from oxford_pet import load_dataset
from models.unet import UNet
from utils import set_seed

ic.configureOutput(prefix="ic|", includeContext=True)


def test(args):
    ic(args)
    test_data = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = UNet(3, 1).to(args.device)
    model.load_state_dict(torch.load(args.model))

    print("Evaluating the model on the test set")
    avg_loss, avg_dice = evaluate(model, test_loader, args, position=0)
    ic(avg_loss, avg_dice)


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
    parser.add_argument(
        "--model",
        default="best_model.pth",
        help="path to the stored model weoght",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="dataset/oxford-iiit-pet/link",
        help="path to the input data",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=8,
        help="batch size",
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

    return parser


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    out_dir = osp.join("log", args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir = out_dir
    args.seed = set_seed(args.seed)

    delattr(args, "output_dir")

    args.model = osp.join(args.out_dir, args.model)
    ic(args)
    if osp.exists(args.model) is False:
        parser.print_help()
        parser.error(f"model weight not found: {args.model}")
    # OxfordPetDataset.download(args.data_path)
    test(args)
