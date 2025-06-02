from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasetclass import VisualCaptioningDataset
from utils import train_val_model

if __name__ == "__main__":
    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base", use_fast=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    dataset = VisualCaptioningDataset(
        csv_file="generated_labels.csv",
        dataset_dir="../preparation/dataset",
        processor=processor,
    )

    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10
    best_val_loss = float("inf")

    model, metrics = train_val_model(model, train_loader, val_loader, optimizer)

    train_losses = metrics["train_losses"]
    val_losses = metrics["val_losses"]

    np.save("train_losses.npy", train_losses)
    np.save("val_losses.npy", val_losses)

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
