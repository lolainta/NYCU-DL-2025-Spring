import argparse
import datetime
import os.path as osp
from icecream import ic
import torch
from torch.utils.data import DataLoader

from evaluate import evaluate
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResUNet
from utils import set_seed, BCEDiceLoss

ic.configureOutput(prefix="ic|", includeContext=True)


def test(args):
    test_data = load_dataset(args.data_path, "test")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = None
    match args.model:
        case "unet":
            model = UNet(3, 1).to(args.device)
        case "resunet":
            model = ResUNet(3, 1).to(args.device)
        case _:
            raise ValueError("Invalid model name")
    model.load_state_dict(torch.load(args.weight))

    criterion = BCEDiceLoss()

    print("Evaluating the model on the test set")
    avg_loss, avg_dice = evaluate(model, test_loader, criterion, args, position=0)
    ic(avg_loss, avg_dice)


def get_args():
    parser = argparse.ArgumentParser(description="Predict masks from input images")
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
        default=32,
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
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="unet",
        choices=["unet", "resunet"],
        help="model name",
    )
    parser.add_argument(
        "--weight",
        "-w",
        type=str,
        default="best_model.pth",
        help="path to the stored model weoght",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for inference",
    )
    return parser


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    assert osp.exists(args.data_path), f"Data path not found: {args.data_path}"

    args.out_dir = osp.join("log", args.output_dir)
    delattr(args, "output_dir")
    args.seed = set_seed(args.seed)

    args.weight = osp.join(args.out_dir, args.weight)
    ic(args)

    if osp.exists(args.weight) is False:
        parser.print_help()
        parser.error(f"model weight not found: {args.weight}")
    # OxfordPetDataset.download(args.data_path)
    test(args)
