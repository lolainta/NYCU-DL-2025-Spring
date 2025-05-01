import os
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dataset import IclevrDataset
from model import ConditionalDDPM


class Trainer:
    def __init__(self, args):
        self.train_dataset = IclevrDataset(args.dataset, "train")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        self.val_dataset = IclevrDataset(args.dataset, "test")
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        self.criteria = nn.MSELoss()
        self.model = ConditionalDDPM().to(args.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.save_freq = args.save_freq
        self.val_freq = args.val_freq
        self.device = args.device
        self.writer = SummaryWriter()

        self.best_loss: float = float("inf")
        self.epoch = 0

    def save_checkpoint(self, path):
        torch.save(
            {
                "epoch": self.epoch,
                "best_loss": self.best_loss,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        tqdm.write(f"Model saved at {path}")

    def train(self):
        for epoch in trange(self.epochs):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            tqdm.write(f"Epoch {epoch}, Train Loss: {train_loss}")

            if epoch % self.save_freq == 0:
                self.save_checkpoint(
                    os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
                )
                tqdm.write(f"Model saved at epoch {epoch}")

            if epoch % self.val_freq == 0:
                val_loss = self.validate()
                tqdm.write(f"Epoch {epoch}, Validation Loss: {val_loss}")
                self.writer.add_scalar("Loss/validation", val_loss, epoch)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint(os.path.join(self.save_dir, "best_model.pth"))
                    tqdm.write(f"Best model saved at epoch {epoch}")

    def train_one_epoch(self):
        self.model.train()
        train_loss = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch: {self.epoch}", leave=True)
        for i, (img, label) in enumerate(progress_bar):
            batch_size = img.shape[0]
            img, label = img.to(self.device), label.to(self.device)
            noise = torch.randn_like(img)

            timesteps = torch.randint(0, 1000, (batch_size,)).long().to(self.device)
            noisy_x = self.noise_scheduler.add_noise(img, noise, timesteps)  # type: ignore
            output = self.model(noisy_x, timesteps, label)

            loss = self.criteria(output, noise)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())
            progress_bar.set_postfix({"Loss": np.mean(train_loss)})

            self.writer.add_scalar(
                "Loss/train-step", loss.item(), self.epoch * len(self.train_loader) + i
            )

        self.writer.add_scalar("Loss/epoch", np.mean(train_loss), self.epoch)
        return np.mean(train_loss)

    def validate(self) -> float:
        self.model.eval()
        val_loss: list[float] = []
        progress_bar = tqdm(self.val_loader, desc="Validation", leave=True)
        with torch.no_grad():
            for i, (img, label) in enumerate(progress_bar):
                img, label = img.to(self.device), label.to(self.device)
                noise = torch.randn_like(img)

                timesteps = (
                    torch.randint(0, 1000, (img.shape[0],)).long().to(self.device)
                )
                noisy_x = self.noise_scheduler.add_noise(img, noise, timesteps)  # type: ignore
                output = self.model(noisy_x, timesteps, label)
                loss = self.criteria(output, noise)
                val_loss.append(loss.item())
                progress_bar.set_postfix({"Loss": np.mean(val_loss)})
                self.writer.add_scalar(
                    "Loss/validation-step",
                    loss.item(),
                    self.epoch * len(self.val_loader) + i,
                )
        assert len(val_loss) > 0, "Validation loss is empty"
        return sum(val_loss) / len(val_loss)


def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--num_workers", type=int, default=20)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--val_freq", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = arg_parser()
    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
