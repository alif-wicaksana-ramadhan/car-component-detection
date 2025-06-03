from tqdm import tqdm
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
import torch
import gc


def train_val_model(
    model: Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: lr_scheduler = None,
    num_epochs: int = 20,
    device: torch.device = "cuda",
    save_path: str = "best_model.pth",
):
    model = model.to(device)
    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]", leave=False
        )
        for batch in train_pbar:
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

            loss = outputs.loss
            train_loss += loss.item()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            del input_ids, pixel_values, attention_mask, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False
        )
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )

                val_loss += outputs.loss.item()

                val_pbar.set_postfix({"loss": f"{outputs.loss.item():.4f}"})

                del input_ids, pixel_values, attention_mask, outputs

        current_val_loss = val_loss / len(val_loader)
        val_losses.append(current_val_loss)

        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            model.save_pretrained("finetuned_blip_best")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                save_path,
            )
            print(f"✓ New best model saved! Val Loss: {best_val_loss:.4f}")

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {current_val_loss:.4f} (Best: {best_val_loss:.4f})")

        if scheduler is not None:
            scheduler.step()

    return model, {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
