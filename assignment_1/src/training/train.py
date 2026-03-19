# Shared train loop for all 3 tasks. No task-specific logic here.

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torch.amp is the current API; torch.cuda.amp still works but is deprecated in 2.4+
from torch.amp import GradScaler, autocast

from ..evaluation.metrics import compute_macro_f1


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    """One full pass over the training loader. Returns average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in loader: # batches of (images, labels), already shuffled by the DataLoader
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # If we pass a scaler, so we'll use AMP - Automatic Mixed Precision - uses both 16-bit and 32-bit floating point numbers to speed up training and reduce memory usage on GPUs that support it
            with autocast(device_type=device.type): # Temporarily enables mixed precision for the operations inside the block
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run the model on loader. Returns {"loss": float, "acc": float, "macro_f1": float}."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    macro_f1 = compute_macro_f1(all_labels, all_preds)

    return {"loss": avg_loss, "acc": accuracy, "macro_f1": macro_f1}
