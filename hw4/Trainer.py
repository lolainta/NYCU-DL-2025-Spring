import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from modules import (
    Generator,
    Gaussian_Predictor,
    Decoder_Fusion,
    Label_Encoder,
    RGB_Encoder,
)

from dataloader import Dataset_Dance
import random
import torch.optim as optim

from tqdm import tqdm, trange

from math import log10

from loguru import logger


def Generate_PSNR(imgs1, imgs2, data_range=1.0):
    """PSNR for torch tensor"""
    # logger.debug(f"imgs1: {imgs1.shape}, imgs2: {imgs2.shape}") (B, C, H, W)
    mse = torch.mean((imgs1 - imgs2) ** 2, dim=(1, 2, 3)).mean()
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.anneal_type = args.kl_anneal_type
        self.step = current_epoch

        match self.anneal_type:
            case "constant":
                self.L = torch.ones(args.num_epoch)
            case "linear":
                self.L = torch.linspace(0, 1, args.num_epoch)
            case "cyclical":
                self.L = self.frange_cycle_linear(
                    args.num_epoch,
                    start=0.0,
                    stop=1.0,
                    n_cycle=args.kl_anneal_cycle,
                    ratio=args.kl_anneal_ratio,
                )
        self.L = self.L.to(args.device)

    def update(self):
        self.step += 1

    def get_beta(self):
        return self.L[self.step]

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
        L = torch.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L


class VAE_Model(nn.Module):
    def __init__(self, args, writer=True):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim
        )
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim
        )

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[2, 5], gamma=0.1
        )
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

        self.writer = SummaryWriter(f"runs/{args.save_root}")
        self.writer.add_text("args", str(args))

        self.best_psnr = 0.0

    def training_stage(self):

        cnt = 0
        for _ in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing: bool = True if random.random() < self.tfr else False

            for img, label in (pbar := tqdm(train_loader, dynamic_ncols=True)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss, (mse, kl, psnr) = self.training_one_step(
                    img, label, adapt_TeacherForcing
                )

                self.writer.add_scalar("MSE/train-step", mse, cnt)
                self.writer.add_scalar("KL/train-step", kl, cnt)
                self.writer.add_scalar("PSNR/train-step", psnr, cnt)
                cnt += 1

                beta = self.kl_annealing.get_beta()
                self.tqdm_bar(
                    f"train [TF: {adapt_TeacherForcing}, {self.tfr:.1f}], beta: {beta:.2f}",
                    pbar,
                    loss.detach().cpu(),
                    mse=mse,
                    kl=kl,
                    psnr=psnr,
                    lr=self.scheduler.get_last_lr()[0],
                )

            if (self.current_epoch + 1) % self.args.per_save == 0:
                self.save(
                    os.path.join(
                        self.args.save_root, f"epoch={self.current_epoch}.ckpt"
                    )
                )

            self.writer.add_scalar("TFR/train", self.tfr, self.current_epoch)
            self.writer.add_scalar(
                "Beta/train", self.kl_annealing.get_beta(), self.current_epoch
            )

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

    @torch.no_grad()
    def eval(self):  # type: ignore
        val_loader = self.val_dataloader()
        assert len(val_loader) == 1, "Ønly one video in val dataloader"

        img = val_loader.dataset[0][0].unsqueeze(0).to(self.args.device)
        label = val_loader.dataset[0][1].unsqueeze(0).to(self.args.device)
        loss, (mse, kl, psnr) = self.val_one_step(img, label)

        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.save(os.path.join(self.args.save_root, f"best.ckpt"))
            logger.info(f"Best PSNR: {psnr}, epoch: {self.current_epoch}")

        self.writer.add_scalar("MSE/val", mse, self.current_epoch)
        self.writer.add_scalar("KL/val", kl, self.current_epoch)
        self.writer.add_scalar("PSNR/val", psnr, self.current_epoch)
        self.writer.add_scalar("Loss/val", loss, self.current_epoch)

    def training_one_step(
        self, img, label, adapt_TeacherForcing: bool
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        # image shape: [batch_size, video_len, channel, height, width]
        # label shape: [batch_size, video_len, channel, height, width]

        mse_loss = torch.zeros(1).to(self.args.device)
        kl_loss = torch.zeros(1).to(self.args.device)
        psnr = torch.zeros(1).to(self.args.device)

        prev_frame = img[:, 0, :, :, :].clone()
        for i in range(1, self.args.train_vi_len):
            trans_prev_frame = self.frame_transformation(prev_frame)
            trans_cur_frame = self.frame_transformation(img[:, i, :, :, :])
            trans_cur_label = self.label_transformation(label[:, i, :, :, :])
            z, mu, logvar = self.Gaussian_Predictor(trans_cur_frame, trans_cur_label)
            df_out = self.Decoder_Fusion(trans_prev_frame, trans_cur_label, z)
            pred_frame = self.Generator(df_out)

            if torch.isnan(pred_frame).any():
                logger.warning(f"Nan in prev_frame")
                break

            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            mse_loss += self.mse_criterion(pred_frame, img[:, i, :, :, :])
            psnr += Generate_PSNR(pred_frame, img[:, i, :, :, :], data_range=1.0)

            prev_frame = img[:, i, :, :, :] if adapt_TeacherForcing else pred_frame

        mse_loss /= self.args.train_vi_len - 1
        kl_loss /= self.args.train_vi_len - 1
        psnr /= self.args.train_vi_len - 1
        loss = mse_loss + self.kl_annealing.get_beta() * kl_loss

        self.optim.zero_grad()
        loss.backward()
        self.optimizer_step()

        return loss, (mse_loss.item(), kl_loss.item(), psnr.item())

    @torch.no_grad()
    def val_one_step(
        self, img, label
    ) -> tuple[torch.Tensor, tuple[float, float, float]]:
        mse_loss = torch.zeros(1).to(self.args.device)
        kl_loss = torch.zeros(1).to(self.args.device)
        psnr = torch.zeros(1).to(self.args.device)
        decoded_frames = [img[:, 0, :, :, :].clone()]

        prev_frame = img[:, 0, :, :, :].clone()
        for i in (pbar := trange(1, self.args.val_vi_len)):
            trans_prev_frame = self.frame_transformation(prev_frame)
            trans_cur_frame = self.frame_transformation(img[:, i, :, :, :])
            trans_cur_label = self.label_transformation(label[:, i, :, :, :])
            z, mu, logvar = self.Gaussian_Predictor(trans_cur_frame, trans_cur_label)
            df_out = self.Decoder_Fusion(trans_prev_frame, trans_cur_label, z)
            pred_frame = self.Generator(df_out)

            decoded_frames.append(pred_frame)

            kl_loss += kl_criterion(mu, logvar, self.batch_size)
            mse_loss += self.mse_criterion(pred_frame, img[:, i, :, :, :])
            psnr += Generate_PSNR(pred_frame, img[:, i, :, :, :], data_range=1.0)

            prev_frame = pred_frame

            self.tqdm_bar(
                f"val [TF: False, {self.tfr:.1f}], beta: {self.kl_annealing.get_beta():.2f}",
                pbar,
                mse_loss.detach().cpu(),
                mse=mse_loss,
                kl=kl_loss,
                psnr=psnr / (i + 1),
                lr=self.scheduler.get_last_lr()[0],
            )

        mse_loss /= self.args.val_vi_len - 1
        kl_loss /= self.args.val_vi_len - 1
        psnr /= self.args.val_vi_len - 1
        loss = mse_loss + self.kl_annealing.get_beta() * kl_loss

        decoded_frames = torch.stack(decoded_frames, dim=1)

        if self.args.store_visualization:
            # if (self.current_epoch + 1) % self.args.per_save == 0:
            self.make_gif(
                decoded_frames[0],
                f"{self.args.save_root}/val_{self.current_epoch}_decoded.gif",
            )

        self.writer.add_scalar("MSE/val", mse_loss.item(), self.current_epoch)
        self.writer.add_scalar("KL/val", kl_loss.item(), self.current_epoch)
        self.writer.add_scalar("PSNR/val", psnr.item(), self.current_epoch)
        self.writer.add_scalar("Loss/val", loss.item(), self.current_epoch)

        return loss, (mse_loss.item(), kl_loss.item(), psnr.item())

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(
            img_name,
            format="GIF",
            append_images=new_list,
            save_all=True,
            duration=40,
            loop=0,
        )

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )

        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="train",
            video_len=self.train_vi_len,
            partial=args.fast_partial if self.args.fast_train else args.partial,
        )
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((self.args.frame_H, self.args.frame_W)),
                transforms.ToTensor(),
            ]
        )
        dataset = Dataset_Dance(
            root=self.args.DR,
            transform=transform,
            mode="val",
            video_len=self.val_vi_len,
            partial=1.0,
        )
        val_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            drop_last=True,
            shuffle=False,
        )
        return val_loader

    def teacher_forcing_ratio_update(self):
        if self.tfr > 0.0:
            if self.current_epoch >= self.tfr_sde:
                self.tfr -= self.tfr_d_step
                if self.tfr < 0.0:
                    self.tfr = 0.0
            else:
                self.tfr = self.args.tfr
        else:
            self.tfr = 0.0
        self.tfr = max(self.tfr, 0.0)
        self.tfr = min(self.tfr, 1.0)
        return self.tfr

    def tqdm_bar(
        self,
        mode,
        pbar,
        loss,
        mse,
        kl,
        psnr,
        lr,
    ):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr:.2e}", refresh=False
        )
        pbar.set_postfix(
            loss=float(loss), mse=float(mse), kl=float(kl), psnr=float(psnr)
        )
        pbar.refresh()

    def save(self, path):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "optimizer": self.state_dict(),
                "lr": self.scheduler.get_last_lr()[0],
                "tfr": self.tfr,
                "last_epoch": self.current_epoch,
            },
            path,
        )
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint["state_dict"], strict=True)
            self.args.lr = checkpoint["lr"]
            self.tfr = checkpoint["tfr"]

            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=[2, 4], gamma=0.1
            )
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint["last_epoch"]
            )
            self.current_epoch = checkpoint["last_epoch"]

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()


def main(args):
    os.makedirs(args.save_root, exist_ok=True)
    with open(os.path.join(args.save_root, "config.txt"), "w") as f:
        f.write(str(args))
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--optim", type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--test", action="store_true")
    parser.add_argument(
        "--store_visualization",
        action="store_true",
        help="If you want to see the result while training",
    )
    parser.add_argument("--DR", type=str, required=True, help="Your Dataset Path")
    parser.add_argument(
        "--save_root", type=str, required=True, help="The path to save your data"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--num_epoch", type=int, default=200, help="number of total epoch"
    )
    parser.add_argument(
        "--per_save", type=int, default=10, help="Save checkpoint every seted epoch"
    )
    parser.add_argument(
        "--partial",
        type=float,
        default=1.0,
        help="Part of the training dataset to be trained",
    )
    parser.add_argument(
        "--train_vi_len", type=int, default=16, help="Training video length"
    )
    parser.add_argument(
        "--val_vi_len", type=int, default=630, help="valdation video length"
    )
    parser.add_argument(
        "--frame_H", type=int, default=32, help="Height input image to be resize"
    )
    parser.add_argument(
        "--frame_W", type=int, default=64, help="Width input image to be resize"
    )

    # Module parameters setting
    parser.add_argument(
        "--F_dim", type=int, default=128, help="Dimension of feature human frame"
    )
    parser.add_argument(
        "--L_dim", type=int, default=32, help="Dimension of feature label frame"
    )
    parser.add_argument("--N_dim", type=int, default=12, help="Dimension of the Noise")
    parser.add_argument(
        "--D_out_dim",
        type=int,
        default=192,
        help="Dimension of the output in Decoder_Fusion",
    )

    # Teacher Forcing strategy
    parser.add_argument(
        "--tfr", type=float, default=1.0, help="The initial teacher forcing ratio"
    )
    parser.add_argument(
        "--tfr_sde",
        type=int,
        default=10,
        help="The epoch that teacher forcing ratio start to decay",
    )
    parser.add_argument(
        "--tfr_d_step",
        type=float,
        default=0.1,
        help="Decay step that teacher forcing ratio adopted",
    )
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="The path of your checkpoints"
    )

    # Training Strategy
    parser.add_argument("--fast_train", action="store_true")
    parser.add_argument(
        "--fast_partial",
        type=float,
        default=0.4,
        help="Use part of the training data to fasten the convergence",
    )
    parser.add_argument(
        "--fast_train_epoch",
        type=int,
        default=5,
        help="Number of epoch to use fast train mode",
    )

    # Kl annealing stratedy arguments
    parser.add_argument(
        "--kl_anneal_type",
        type=str,
        default="cyclical",
        choices=["cyclical", "linear", "constant"],
        help="Type of KL annealing strategy",
    )
    parser.add_argument("--kl_anneal_cycle", type=int, default=10, help="")
    parser.add_argument("--kl_anneal_ratio", type=float, default=1, help="")

    args = parser.parse_args()

    logger.info(f"{args}")
    main(args)
