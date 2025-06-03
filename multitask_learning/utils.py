from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from sklearn.metrics import accuracy_score
import numpy as np
import torch


def train_val_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler: lr_scheduler = None,
    num_epochs: int = 20,
    device: torch.device = "cuda",
    save_path: str = "best_model.pth",
):
    model = model.to(device)
    train_task_accuracies = []
    train_losses = []
    val_task_accuracies = []
    val_losses = []

    # Initialize best validation loss to infinity
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_preds, train_true = [], []
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False
        )
        for images, labels in train_pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss, metric = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = (outputs > 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_true.extend(labels.cpu().numpy())
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_preds = np.array(train_preds)
        train_true = np.array(train_true)
        accuracies = [
            accuracy_score(train_true[:, i], train_preds[:, i]) for i in range(5)
        ]
        train_task_accuracies.append(accuracies)
        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_true = [], []
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False
        )
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss, metric = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(labels.cpu().numpy())
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        accuracies = [accuracy_score(val_true[:, i], val_preds[:, i]) for i in range(5)]

        val_task_accuracies.append(accuracies)
        current_val_loss = val_loss / len(val_loader)
        val_losses.append(current_val_loss)

        # Save model if validation loss improved
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracies": accuracies,
                },
                save_path,
            )
            print(f"✓ New best model saved! Val Loss: {best_val_loss:.4f}")

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {current_val_loss:.4f} (Best: {best_val_loss:.4f})")
        print(
            f"Val Accuracies: {dict(zip(['rear_right', 'rear_left', 'front_right', 'front_left', 'hood'], accuracies))}"
        )

        if scheduler is not None:
            scheduler.step()

    return model, {
        "train_task_accuracies": train_task_accuracies,
        "train_losses": train_losses,
        "val_task_accuracies": val_task_accuracies,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }


def test_model(model: Module, test_loader: DataLoader, device: torch.device = "cuda"):
    model = model.to(device)
    model.eval()

    test_preds, test_true = [], []
    test_pbar = tqdm(test_loader, desc="[Test]", leave=False)
    with torch.no_grad():
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(labels.cpu().numpy())
            acc = accuracy_score(labels.cpu().numpy(), preds)
            test_pbar.set_postfix({"acc": f"{acc:.4f}"})

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    accuracies = [accuracy_score(test_true[:, i], test_preds[:, i]) for i in range(5)]

    print(
        f"Test Accuracies: {dict(zip(['rear_right', 'rear_left', 'front_right', 'front_left', 'hood'], accuracies))}"
    )


def calculate_mean_std_efficient(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std
