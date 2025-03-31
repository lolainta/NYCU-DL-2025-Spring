import argparse
from loguru import logger
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import yaml

from models import MaskGit as VQGANTransformer
from utils import LoadTrainData


class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(
            device=args.device
        )
        self.optim, self.scheduler = self.configure_optimizers()
        self.prepare_training()
        self.writer = SummaryWriter()

        train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=True,
        )
        val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
        )

    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = []
        for i, data in enumerate(
            tqdm(self.train_loader, position=0, leave=True, unit_scale=args.batch_size)
        ):
            self.optim.zero_grad()
            data = data.to(args.device)
            logits, z_indices = self.model(data)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
            loss.backward()
            losses.append(loss.item())
            if i % args.accum_grad == 0:
                self.optim.step()
            self.writer.add_scalar("Loss", loss.item(), i)
        self.writer.add_scalar("Epoch Loss", np.mean(losses), epoch)
        logger.info(f"Epoch {epoch} Loss: {np.mean(losses)}")
        return np.mean(losses)

    def eval_one_epoch(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in tqdm(
                self.val_loader,
                position=0,
                leave=True,
                unit_scale=args.batch_size,
            ):
                data = data.to(args.device)
                logits, z_indices = self.model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), z_indices.view(-1)
                )
                losses.append(loss.item())
        self.writer.add_scalar("Val Loss", np.mean(losses), epoch)
        logger.info(f"Epoch {epoch} Val Loss: {np.mean(losses)}")
        return np.mean(losses)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.96),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return optimizer, scheduler


def get_args():
    parser = argparse.ArgumentParser(description="MaskGIT")
    parser.add_argument(
        "--train_d_path",
        type=str,
        default="./dataset/train/",
        help="Training Dataset Path",
    )
    parser.add_argument(
        "--val_d_path",
        type=str,
        default="./dataset/val/",
        help="Validation Dataset Path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Which device the training is on.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Number of epochs to train (default: 50)",
    )
    parser.add_argument(
        "--accum-grad",
        type=int,
        default=1,
        help="Number for gradient accumulation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
        help="Learning rate.",
    )
    parser.add_argument(
        "--MaskGitConfig",
        type=str,
        default="config/MaskGit.yml",
        help="Configurations for TransformerVQGAN",
    )
    args = parser.parse_args()
    return args


def main(args):
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    best_train = np.inf
    best_val = np.inf

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_transformer.train_one_epoch(epoch)
        val_loss = train_transformer.eval_one_epoch(epoch)

        if train_loss < best_train:
            best_train = train_loss
            torch.save(
                train_transformer.model.transformer.state_dict(),
                f"transformer_checkpoints/best_train_ckpt.pt",
            )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                train_transformer.model.transformer.state_dict(),
                f"transformer_checkpoints/best_val_ckpt.pt",
            )


if __name__ == "__main__":
    args = get_args()
    logger.info(args)
    main(args)
