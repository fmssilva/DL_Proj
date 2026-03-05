# Data pipeline: transforms, Dataset class, class weights, DataLoaders.
# Everything downstream imports from here.

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from ..config import CLASSES, NUM_CLASSES, SEED


print("HELLO")


# ── transforms ────────────────────────────────────────────────────────────────

# ImageNet mean/std — good default even for non-ImageNet data; revisit if EDA shows big drift
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Grayscale normalization: mean/std for a single channel (computed empirically;
# 0.5/0.5 is a safe neutral default when dataset-specific stats aren't available)
_GRAY_MEAN = [0.5]
_GRAY_STD  = [0.5]


def get_base_transforms(size: int) -> transforms.Compose:
    """RGB pipeline: resize, ToTensor (scales to [0,1]), normalize with ImageNet constants."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_augment_transforms(size: int) -> transforms.Compose:
    """RGB pipeline with augmentation: flip, color jitter, small rotation. For CNN/Transfer training."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_gray_transforms(size: int, equalize: bool = False) -> transforms.Compose:
    """
    Grayscale pipeline: resize, optional histogram equalization, convert to 1-channel,
    ToTensor, normalize. Outputs (1, H, W) tensors — input_dim = size*size*1.
    equalize=True applies PIL histogram equalization to stretch contrast before grayscaling.
    """
    steps = [transforms.Resize((size, size))]
    if equalize:
        # apply histogram equalization on the RGB image before converting to gray
        steps.append(transforms.Lambda(lambda img: ImageOps.equalize(img)))
    steps += [
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_GRAY_MEAN, std=_GRAY_STD),
    ]
    return transforms.Compose(steps)


def get_gray_aug_transforms(size: int, equalize: bool = False) -> transforms.Compose:
    """Grayscale pipeline with augmentation. No color jitter (no color info in gray)."""
    steps = [transforms.Resize((size, size))]
    if equalize:
        steps.append(transforms.Lambda(lambda img: ImageOps.equalize(img)))
    steps += [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_GRAY_MEAN, std=_GRAY_STD),
    ]
    return transforms.Compose(steps)


# ── dataset ───────────────────────────────────────────────────────────────────

class PokemonDataset(Dataset):
    """
    PokemonDataset Class: 
    Loads images and labels for training or inference.
    Modes:
     - Training: Loads images and their labels from a CSV file or a pre-loaded DataFrame.
     - Inference: If csv_path is None and df is None, loads all PNGs from a directory and returns their filenames (UUID stems) instead of labels.
    """

    def __init__(
        self,
        img_dir: Path,
        transform: transforms.Compose,
        csv_path: Optional[Path] = None,
        indices: Optional[list] = None,
        df: Optional[pd.DataFrame] = None,
    ):
        self.img_dir   = Path(img_dir)
        self.transform = transform

        if csv_path is None and df is None:
            # inference mode — no labels, just iterate the test folder
            self._paths  = sorted(self.img_dir.glob("*.png"))
            self._labels = None
        else:
            # training mode — use provided df or read from csv_path
            data = df if df is not None else pd.read_csv(csv_path)
            # label string -> int using sorted CLASSES (index is the integer label)
            label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
            self._paths  = [self.img_dir / f"{row.Id}.png" for row in data.itertuples()]
            self._labels = [label_to_idx[row.label] for row in data.itertuples()]

            # support subsetting for stratified splits
            if indices is not None:
                self._paths  = [self._paths[i]  for i in indices]
                self._labels = [self._labels[i] for i in indices]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        """ Loads an image, applies transforms, and returns either (tensor, label) or (tensor, uuid_stem) depending on mode."""
        img = Image.open(self._paths[idx]).convert("RGB")
        tensor = self.transform(img)

        if self._labels is None:
            # inference mode — return uuid string so submission.py can write the Id column
            return tensor, self._paths[idx].stem

        return tensor, self._labels[idx]


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(labels: list) -> torch.Tensor:
    """
    Inverse-frequency weights: total / (NUM_CLASSES * class_count).
    Rarer classes (Ground, Rock) get higher weight -> weighted CrossEntropyLoss.
    """
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    total  = len(labels)
    weights = total / (NUM_CLASSES * counts)
    return torch.tensor(weights, dtype=torch.float32)


# ── loaders ───────────────────────────────────────────────────────────────────

def get_train_val_loaders(
    csv_path: Path,
    img_dir: Path,
    img_size: int,
    batch_size: int,
    augment: bool = False,
    use_sampler: bool = False,
    num_workers: int = 2,
    df_override: Optional[pd.DataFrame] = None,
    grayscale: bool = False,
    equalize: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """
    Stratified 80/20 split -> two DataLoaders.
    augment=True: training loader uses augmentation transforms.
    use_sampler=True: adds WeightedRandomSampler to training loader.
    grayscale=True: outputs (1, H, W) tensors instead of (3, H, W). input_dim = size*size.
    equalize=True: applies histogram equalization before grayscale (only used when grayscale=True).
    df_override: pass a pre-filtered DataFrame instead of reading csv_path (used for FAST_RUN).
    """
    df = df_override if df_override is not None else pd.read_csv(csv_path)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_labels   = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices  = list(range(len(df)))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SEED,
        stratify=all_labels,
    )

    # pick the right transform family (RGB vs gray, base vs augmented)
    if grayscale:
        train_transform = get_gray_aug_transforms(img_size, equalize=equalize) if augment else get_gray_transforms(img_size, equalize=equalize)
        val_transform   = get_gray_transforms(img_size, equalize=equalize)
    else:
        train_transform = get_augment_transforms(img_size) if augment else get_base_transforms(img_size)
        val_transform   = get_base_transforms(img_size)

    # pass df directly so PokemonDataset uses the (possibly subsampled) rows, not the full CSV
    train_ds = PokemonDataset(img_dir, train_transform, df=df, indices=train_idx)
    val_ds   = PokemonDataset(img_dir, val_transform,   df=df, indices=val_idx)

    # build sampler only for train loader when requested
    sampler       = None
    train_shuffle = True
    if use_sampler:
        train_labels  = [all_labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels)
        sample_weights = [class_weights[lbl].item() for lbl in train_labels]
        sampler       = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_shuffle = False  # sampler and shuffle are mutually exclusive in DataLoader

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=train_shuffle,
        sampler=sampler, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader
