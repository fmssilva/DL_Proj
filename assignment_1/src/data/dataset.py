# Data pipeline: transforms, Dataset class, class weights, DataLoaders.
# Everything downstream imports from here.

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.config import CLASSES, NUM_CLASSES, SEED


# ── transforms ────────────────────────────────────────────────────────────────

# ImageNet mean/std — good default even for non-ImageNet data; revisit if EDA shows big drift
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_base_transforms(size: int) -> transforms.Compose:
    """Returns a pipeline that resizes images, converts them to tensors, and normalizes them using the constants above. 
    Used for validation, testing, and MLP training."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def get_augment_transforms(size: int) -> transforms.Compose:
    """Adds random horizontal flips, color jitter, and small rotations for data augmentation. 
    Used for training CNNs or transfer learning."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ── dataset ───────────────────────────────────────────────────────────────────

class PokemonDataset(Dataset):
    """
    PokemonDataset Class: 
    Loads images and labels for training or inference.
    Modes:
     - Training: Loads images and their labels from a CSV file.
     - Inference: If csv_path is None, loads all PNGs from a directory and returns their filenames (UUID stems) instead of labels.
    """

    def __init__(
        self,
        img_dir: Path,
        transform: transforms.Compose,
        csv_path: Optional[Path] = None,
        indices: Optional[list] = None,
    ):
        self.img_dir   = Path(img_dir)
        self.transform = transform

        if csv_path is None:
            # inference mode — no labels, just iterate the test folder
            self._paths  = sorted(self.img_dir.glob("*.png"))
            self._labels = None
        else:
            df = pd.read_csv(csv_path)
            # label string -> int using sorted CLASSES (index is the integer label)
            label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
            self._paths  = [self.img_dir / f"{row.Id}.png" for row in df.itertuples()]
            self._labels = [label_to_idx[row.label] for row in df.itertuples()]

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
) -> tuple[DataLoader, DataLoader]:
    """
    Stratified 80/20 split -> two DataLoaders.
    augment=True uses get_augment_transforms for the train loader (val always uses base).
    use_sampler=True adds WeightedRandomSampler to the train loader.
    """
    df = pd.read_csv(csv_path)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_labels   = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices  = list(range(len(df)))

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=SEED,
        stratify=all_labels,
    )

    train_transform = get_augment_transforms(img_size) if augment else get_base_transforms(img_size)
    val_transform   = get_base_transforms(img_size)

    train_ds = PokemonDataset(img_dir, train_transform, csv_path, indices=train_idx)
    val_ds   = PokemonDataset(img_dir, val_transform,   csv_path, indices=val_idx)

    # build sampler only for train loader when requested
    sampler      = None
    train_shuffle = True
    if use_sampler:
        train_labels  = [all_labels[i] for i in train_idx]
        class_weights = compute_class_weights(train_labels)
        # each sample gets the weight of its class
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
