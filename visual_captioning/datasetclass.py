from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class VisualCaptioningDataset(Dataset):
    def __init__(
        self, csv_file: str, dataset_dir: str, processor, max_length: int = 128
    ):
        self.processor = processor
        self.max_length = max_length
        self.dataset_dir = dataset_dir
        df = pd.read_csv(csv_file)
        self.image_paths = df["file_name"]
        self.captions = df["caption"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths.iloc[idx]
        image = Image.open(f"{self.dataset_dir}/{image_path}").convert("RGB")

        caption = str(self.captions.iloc[idx])

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoding = {k: v.squeeze() for k, v in encoding.items()}

        return encoding
