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
    """
    One full pass over the training loader.
    scaler != None enables AMP (Task 3 only); None = normal FP32 (Tasks 1 & 2).
    Returns average loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            # AMP path — halves memory, ~2x faster on supported GPUs
            # device_type required by torch.amp.autocast (new API in 2.4+)
            with autocast(device_type=device.type):
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
    """
    Run the model on loader and return loss, accuracy, and macro F1.
    Returns: {"loss": float, "acc": float, "macro_f1": float}
    """
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

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


def run_epoch(
    epoch: int,
    total_epochs: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> tuple[float, dict]:
    """
    Convenience wrapper: train + evaluate + print one-line log.
    Returns (train_loss, val_metrics_dict).
    """
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
    val_metrics = evaluate(model, val_loader, criterion, device)
    elapsed = time.time() - t0

    print(
        f"Epoch {epoch}/{total_epochs} | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_metrics['loss']:.4f} | "
        f"val_f1={val_metrics['macro_f1']:.4f} | "
        f"time={elapsed:.1f}s"
    )

    return train_loss, val_metrics
