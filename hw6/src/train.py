import os
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm, trange
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from dataset import IclevrDataset
from model import ConditionalDDPM


class Trainer:
    def __init__(self, args):
        self.train_loader = DataLoader(
            IclevrDataset(args.dataset, "train"),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        self.val_loader = DataLoader(
            IclevrDataset(args.dataset, "test"),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        self.criteria = nn.MSELoss()
        self.model = ConditionalDDPM().to(args.device)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.epochs = args.epochs
        self.save_dir = args.save_dir
        self.save_ckpt_period = args.save_ckpt_period
        self.save_img_period = args.save_img_period
        self.device = args.device
        self.writer = SummaryWriter()

        self.best_loss: float = float("inf")
        self.epoch = 0

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "imgs"), exist_ok=True)

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
        for epoch in trange(
            self.epochs, desc="Training", position=0, dynamic_ncols=True
        ):
            self.epoch = epoch
            train_loss = self.train_one_epoch()
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            tqdm.write(f"Epoch {epoch}, Train Loss: {train_loss}")

            if (epoch + 1) % self.save_ckpt_period == 0:
                self.save_checkpoint(
                    os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
                )

            if (epoch + 1) % self.save_img_period == 0:
                self.save_images()

    def train_one_epoch(self):
        self.model.train()
        train_loss = []
        for i, (img, label) in enumerate(
            tqdm(
                self.train_loader,
                desc=f"Epoch: {self.epoch}",
                leave=True,
                position=1,
                dynamic_ncols=True,
            )
        ):
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

            self.writer.add_scalar(
                "Loss/train-step", loss.item(), self.epoch * len(self.train_loader) + i
            )

        self.writer.add_scalar("Loss/epoch", np.mean(train_loss), self.epoch)
        return np.mean(train_loss)

    def save_images(self):
        self.model.eval()

        for idx, (y, label) in enumerate(
            tqdm(
                self.val_loader,
                desc=f"Epoch: {self.epoch}",
                leave=True,
                position=1,
                dynamic_ncols=True,
            )
        ):
            tqdm.write(f"Image {idx}: {label}")
            y = y.to(self.device)
            x = torch.randn(1, 3, 64, 64).to(self.device)
            denoising_result = []
            for i, t in enumerate(self.noise_scheduler.timesteps):
                with torch.no_grad():
                    residual = self.model(x, t, y)

                x = self.noise_scheduler.step(residual, t, x).prev_sample  # type: ignore
                if i % (len(self.noise_scheduler.timesteps) // 10) == 0:
                    denoising_result.append(x.squeeze(0))

            denoising_result.append(x.squeeze(0))
            denoising_result = torch.stack(denoising_result)
            row_image = make_grid(
                (denoising_result + 1) / 2, nrow=denoising_result.shape[0], pad_value=0
            )
            save_image(
                row_image,
                os.path.join(self.save_dir, "imgs", f"ep{self.epoch}_{idx}.png"),
            )
            tqdm.write(f"Saved image for epoch {self.epoch}, index {i}")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--num_workers", type=int, default=20)

    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_ckpt_period", type=int, default=10)
    parser.add_argument("--save_img_period", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
