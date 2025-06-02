from torch.utils.data import DataLoader
from torchvision import transforms

from datasetclass import MultiTaskDataset
from utils import calculate_mean_std_efficient

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

dataset = MultiTaskDataset(
    csv_file="../preparation/dataset/metadata.csv",
    root_dir="../preparation/dataset",
    transform=transform,
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

mean, std = calculate_mean_std_efficient(dataloader)

print(mean, std)
