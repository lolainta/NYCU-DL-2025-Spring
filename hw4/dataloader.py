import os
from glob import glob
import torch
from torch import stack
from torchvision.transforms import v2
from torch.utils.data import Dataset as torchData

from torchvision.datasets.folder import default_loader as imgloader
from torch import stack


def get_key(fp):
    filename = fp.split("/")[-1]
    filename = filename.split(".")[0].replace("frame", "")
    return int(filename)


class Dataset_Dance(torchData):
    """
    Args:
        root (str)      : The path of your Dataset
        transform       : Transformation to your dataset
        mode (str)      : train, val
        partial (float) : Percentage of your Dataset, may set to use part of the dataset
    """

    def __init__(self, root, transform, mode="train", video_len=7, partial=1.0):
        super().__init__()
        assert mode in ["train", "val"], "There is no such mode !!!"
        if mode == "train":
            self.img_folder = sorted(
                glob(os.path.join(root, "train/train_img/*.png")), key=get_key
            )
            self.prefix = "train"
        elif mode == "val":
            self.img_folder = sorted(
                glob(os.path.join(root, "val/val_img/*.png")), key=get_key
            )
            self.prefix = "val"
        else:
            raise ValueError("There is no such mode !!!")

        self.transform = transform
        self.partial = partial
        self.video_len = video_len

        self.to_tensor = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __len__(self):
        return int(len(self.img_folder) * self.partial) // self.video_len

    def __getitem__(self, index):
        path = self.img_folder[index]
        imgs = []
        labels = []
        for i in range(self.video_len):
            label_list = self.img_folder[(index * self.video_len) + i].split("/")
            label_list[-2] = self.prefix + "_label"

            img_name = self.img_folder[(index * self.video_len) + i]
            label_name = "/".join(label_list)
            img, label = imgloader(img_name), imgloader(label_name)
            img = v2.functional.to_image(img)
            label = v2.functional.to_image(label)

            imgs.append(self.to_tensor(img))
            labels.append(self.to_tensor(label))

        transformed = self.transform(*imgs, *labels)
        imgs = transformed[: self.video_len]
        labels = transformed[self.video_len :]
        imgs = stack(imgs)
        labels = stack(labels)
        return imgs, labels
