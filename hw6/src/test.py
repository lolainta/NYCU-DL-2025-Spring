from argparse import ArgumentParser
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluator import evaluation_model
from dataset import IclevrDataset
from model import ConditionalDDPM
from torchvision.utils import make_grid, save_image


class Tester:
    def __init__(self, args):
        self.test_loader = DataLoader(
            IclevrDataset(args.dataset, "test"),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        self.new_test_loader = DataLoader(
            IclevrDataset(args.dataset, "new_test"),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )
        self.manual_test_loader = DataLoader(
            IclevrDataset(args.dataset, "manual_test"),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
        )

        self.device = args.device
        self.model = ConditionalDDPM().to(self.device)
        self.model.load_state_dict(torch.load(args.ckpt)["model"])
        self.model.eval()

        self.eval_model = evaluation_model()

        self.time_steps = torch.load(args.ckpt)["time_steps"]
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=self.time_steps)

        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "new_test"), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, "manual_test"), exist_ok=True)

    def test(self):
        test_acc = self.inference(
            self.test_loader,
            os.path.join(self.save_dir, "test"),
        )
        new_test_acc = self.inference(
            self.new_test_loader,
            os.path.join(self.save_dir, "new_test"),
        )
        manual_test_acc = self.inference(
            self.manual_test_loader,
            os.path.join(self.save_dir, "manual_test"),
        )
        print(f"Test accuracy: {test_acc:.4f}")
        print(f"New test accuracy: {new_test_acc:.4f}")
        print(f"Manual test accuracy: {manual_test_acc:.4f}")

    def inference(self, loader, save_dir):
        all_results = []
        accs = []
        for idx, (y, label) in enumerate(pbar := tqdm(loader)):
            y = y.to(self.device)
            x = torch.randn(1, 3, 64, 64).to(self.device)
            denoising_results = []
            for i, t in enumerate(self.noise_scheduler.timesteps):
                with torch.no_grad():
                    residual = self.model(x, t, y)

                x = self.noise_scheduler.step(residual, t, x).prev_sample  # type: ignore
                if i % (self.time_steps // 10) == 0:
                    denoising_results.append(x.squeeze(0))
            acc = self.eval_model.eval(x, y)
            tqdm.write(f"image: {idx}, label: {label}, accuracy: {acc:.4f}")
            accs.append(acc)
            pbar.set_postfix_str(f"image: {idx}, accuracy: {np.mean(accs):.4f}")
            pbar.refresh()

            denoising_results.append(x.squeeze(0))
            denoising_results = torch.stack(denoising_results)
            row_image = make_grid(
                (denoising_results + 1) / 2,
                nrow=denoising_results.shape[0],
                pad_value=0,
            )
            save_image(
                row_image,
                os.path.join(save_dir, f"{idx}.png"),
            )
            all_results.append(denoising_results[-1])
        all_results = torch.stack(all_results)
        save_image(
            make_grid(
                (all_results + 1) / 2,
                nrow=8,
                pad_value=0,
            ),
            os.path.join(save_dir, "all_results.png"),
        )
        return np.mean(accs)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="checkpoints/checkpoint_300.pth")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--num_workers", type=int, default=20)

    parser.add_argument("--save_dir", type=str, default="results")
    args = parser.parse_args()
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint {args.ckpt} does not exist")
    return args


def main():
    args = get_args()
    tester = Tester(args)
    tester.test()
    print("Testing completed.")


if __name__ == "__main__":
    main()
