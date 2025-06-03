import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from datasetclass import MultiTaskDataset
from model import MultiTaskModel
from utils import test_model

if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.8220, 0.8310, 0.8322], std=[0.2968, 0.2954, 0.2816]
            ),
        ]
    )

    dataset = MultiTaskDataset(
        csv_file="../preparation/dataset/metadata.csv",
        root_dir="../preparation/dataset",
        transform=transform,
    )

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskModel(num_tasks=5)
    state_dict = torch.load("best_model.pth")
    model.load_state_dict(state_dict["model_state_dict"])

    model = model.to(device)
    model.eval()

    test_model(model, test_loader, device)
