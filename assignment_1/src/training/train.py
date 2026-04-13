# Shared train loop for all 3 tasks. No task-specific logic here.

import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# torch.amp is the current API; torch.cuda.amp still works but is deprecated in 2.4+
from torch.amp import GradScaler, autocast

from ..evaluation.metrics import compute_macro_f1

import numpy as np
import random

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size, _, H, W = x.size()
    index = torch.randperm(batch_size).to(x.device)

    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y_a, y_b, lam


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    use_mixup_cutmix: bool = False,
    alpha: float = 0.4,
) -> float:
    """One full pass over the training loader. Returns average loss."""
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # -----------------------------
        # MixUp / CutMix
        # -----------------------------
        mode = None
        lam = 1.0

        if use_mixup_cutmix:
            r = random.random()
            if r < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels, alpha)
                mode = "mixup"
            else:
                images, y_a, y_b, lam = cutmix_data(images, labels, alpha)
                mode = "cutmix"

        # -----------------------------
        # Forward + backward (AMP aware)
        # -----------------------------
        if scaler is not None:
            with autocast(device_type=device.type):
                logits = model(images)
                if mode in ("mixup", "cutmix"):
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(images)
            if mode in ("mixup", "cutmix"):
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                loss = criterion(logits, labels)

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
