import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def transform_img(img):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transform(img)


class IclevrDataset(Dataset):
    def __init__(self, root: str, mode="train"):
        super().__init__()
        assert mode in [
            "train",
            "test",
            "new_test",
        ], f"mode {mode} not in [train, test, new_test]"
        assert os.path.exists(root), f"{root} does not exist"

        self.root = root
        self.mode = mode

        json_path = os.path.join(root, f"{mode}.json")
        with open(json_path, "r") as json_file:
            self.json_data = json.load(json_file)
            match mode:
                case "train":
                    self.img_paths = list(self.json_data.keys())
                    self.labels = list(self.json_data.values())
                case _:
                    self.labels = list(self.json_data)

        with open(os.path.join(root, "objects.json"), "r") as json_file:
            self.objects_dict = json.load(json_file)
        self.labels_one_hot = torch.zeros(len(self.labels), len(self.objects_dict))

        for i, label in enumerate(self.labels):
            self.labels_one_hot[i][[self.objects_dict[j] for j in label]] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor | list]:
        match self.mode:
            case "train":
                img_path = os.path.join(self.root, "iclevr", self.img_paths[index])
                img = Image.open(img_path).convert("RGB")
                img = transform_img(img)
                label_one_hot = self.labels_one_hot[index]
                return img, label_one_hot
            case _:
                label_one_hot = self.labels_one_hot[index]
                semantic_label = self.labels[index]
                return label_one_hot, semantic_label


if __name__ == "__main__":
    dataset = IclevrDataset(root="dataset", mode="train")
    print(len(dataset))
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    print(x.shape, y.shape)
    dataset = IclevrDataset(root="dataset", mode="test")
    print(len(dataset))
    y, label = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, list)
    print(y.shape, label)
