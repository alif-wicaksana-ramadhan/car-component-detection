from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import torch


class MultiTaskDataset(Dataset):
    def __init__(self, csv_file: str, root_dir: str, transform: transforms = None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        (
            filename,
            rear_right_door,
            rear_left_door,
            front_right_door,
            front_left_door,
            hood,
        ) = self.data.iloc[idx]
        img_path = f"{self.root_dir}/{filename}"

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            [
                rear_right_door,
                rear_left_door,
                front_right_door,
                front_left_door,
                hood,
            ],
            dtype=torch.float32,
        )

        return image, label


if "__main__" == __name__:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = MultiTaskDataset(
        csv_file="car_components_dataset.csv", root_dir="dataset", transform=transform
    )
