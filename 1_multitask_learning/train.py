import torch
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from datasetclass import MultiTaskDataset
from model import MultiTaskModel
from loss_function import UncertaintyWeightedLoss
from utils import train_val_model, test_model

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

    # Split train, val, test
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskModel(num_tasks=5)

    criterion = UncertaintyWeightedLoss(num_tasks=5)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)

    model, metrics = train_val_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=50,
        device=device,
    )

    print("Training complete!")

    test_model(model, test_loader, device)

    train_task_accuracies = metrics["train_task_accuracies"]
    train_losses = metrics["train_losses"]
    val_task_accuracies = metrics["val_task_accuracies"]
    val_losses = metrics["val_losses"]

    # Save to file
    np.save("train_task_accuracies.npy", train_task_accuracies)
    np.save("train_losses.npy", train_losses)
    np.save("val_task_accuracies.npy", val_task_accuracies)
    np.save("val_losses.npy", val_losses)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(train_task_accuracies, label="Train Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    plt.plot(val_task_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
