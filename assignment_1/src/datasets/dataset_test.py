# Local tests for dataset.py — run on CPU before touching Colab.
# Run with: python -m src.datasets.dataset_test  (from assignment_1/ root)

from collections import Counter
from pathlib import Path

from src.config import CLASSES, NUM_CLASSES
from src.datasets.dataset import (
    PokemonDataset,
    compute_class_weights,
    get_augment_transforms,
    get_base_transforms,
    get_gray_transforms,
    get_gray_aug_transforms,
    get_train_val_loaders,
)

# ── paths (relative to assignment_1/) ─────────────────────────────────────────
CSV_PATH  = Path("data/train_labels.csv")
TRAIN_DIR = Path("data/Train")
TEST_DIR  = Path("data/Test")

# test constants — mirror notebook defaults
_IMG_SIZE   = 64
_BATCH_SIZE = 64


def test_dataset_length():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(_IMG_SIZE), csv_path=CSV_PATH)
    assert len(ds) == 3600, f"Expected 3600 samples, got {len(ds)}"
    print(f"[PASS] dataset length: {len(ds)}")


def test_item_shape_and_dtype():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(_IMG_SIZE), csv_path=CSV_PATH)
    tensor, label = ds[0]

    # shape must be (C, H, W) after transform
    assert tensor.shape == (3, _IMG_SIZE, _IMG_SIZE), \
        f"Expected (3,{_IMG_SIZE},{_IMG_SIZE}), got {tensor.shape}"
    assert tensor.dtype == __import__("torch").float32, f"Expected float32, got {tensor.dtype}"

    # after ImageNet normalisation values live roughly in [-3, 3]
    assert tensor.min() > -4.0 and tensor.max() < 4.0, \
        f"Values out of expected range: min={tensor.min():.2f}, max={tensor.max():.2f}"

    print(f"[PASS] item shape {tensor.shape}, dtype {tensor.dtype}, "
          f"range [{tensor.min():.2f}, {tensor.max():.2f}]")


def test_label_range():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(_IMG_SIZE), csv_path=CSV_PATH)
    labels = [ds[i][1] for i in range(len(ds))]
    assert min(labels) == 0 and max(labels) == NUM_CLASSES - 1, \
        f"Labels must be in [0, {NUM_CLASSES-1}], got [{min(labels)}, {max(labels)}]"
    print(f"[PASS] labels in [0, {NUM_CLASSES-1}], unique: {sorted(set(labels))}")


def test_stratified_split():
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, _IMG_SIZE, _BATCH_SIZE, augment=False, use_sampler=False
    )
    train_labels = [lbl.item() for _, lbls in train_loader for lbl in lbls]
    val_labels   = [lbl.item() for _, lbls in val_loader   for lbl in lbls]

    assert len(train_labels) + len(val_labels) == 3600, \
        f"Split sizes don't add up to 3600: {len(train_labels)} + {len(val_labels)}"

    # roughly 80/20
    assert 2800 <= len(train_labels) <= 2900, f"Train size unexpected: {len(train_labels)}"
    assert  700 <= len(val_labels)   <=  800, f"Val   size unexpected: {len(val_labels)}"

    # both splits should contain all 9 classes (stratified guarantees this)
    assert len(set(train_labels)) == NUM_CLASSES, "Train split missing some classes"
    assert len(set(val_labels))   == NUM_CLASSES, "Val split missing some classes"

    train_dist = Counter(train_labels)
    val_dist   = Counter(val_labels)
    print(f"[PASS] train size={len(train_labels)}, val size={len(val_labels)}")
    print(f"       train class counts: {dict(sorted(train_dist.items()))}")
    print(f"       val   class counts: {dict(sorted(val_dist.items()))}")


def test_class_weights():
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    labels = [label_to_idx[lbl] for lbl in df["label"]]

    weights = compute_class_weights(labels)
    assert weights.shape == (NUM_CLASSES,), f"Expected shape ({NUM_CLASSES},), got {weights.shape}"
    assert (weights > 0).all(), "All weights must be positive"

    # Ground (idx 4) and Rock (idx 7) should be heavier than Water (idx 8)
    ground_w = weights[CLASSES.index("Ground")].item()
    rock_w   = weights[CLASSES.index("Rock")].item()
    water_w  = weights[CLASSES.index("Water")].item()
    assert ground_w > water_w, f"Ground weight {ground_w:.3f} should be > Water {water_w:.3f}"
    assert rock_w   > water_w, f"Rock weight {rock_w:.3f} should be > Water {water_w:.3f}"

    print(f"[PASS] class weights shape={weights.shape}, all positive")
    for i, cls in enumerate(CLASSES):
        print(f"       {cls:10s}: {weights[i]:.4f}")


def test_augment_same_shape():
    base_ds  = PokemonDataset(TRAIN_DIR, get_base_transforms(_IMG_SIZE),   csv_path=CSV_PATH)
    aug_ds   = PokemonDataset(TRAIN_DIR, get_augment_transforms(_IMG_SIZE), csv_path=CSV_PATH)
    t_base,  _ = base_ds[0]
    t_aug,   _ = aug_ds[0]
    assert t_base.shape == t_aug.shape, \
        f"Augmented shape {t_aug.shape} != base shape {t_base.shape}"
    print(f"[PASS] augmented output shape matches base: {t_aug.shape}")


def test_inference_mode():
    ds = PokemonDataset(TEST_DIR, get_base_transforms(_IMG_SIZE), csv_path=None)
    assert len(ds) == 900, f"Expected 900 test images, got {len(ds)}"

    tensor, uuid = ds[0]
    assert tensor.shape == (3, _IMG_SIZE, _IMG_SIZE), \
        f"Test tensor shape wrong: {tensor.shape}"
    assert isinstance(uuid, str) and len(uuid) > 0, f"Expected UUID string, got: {uuid!r}"

    print(f"[PASS] inference mode: {len(ds)} images, sample uuid={uuid!r}, shape={tensor.shape}")


def test_dataloader_batch_shapes():
    """DataLoader must yield (B, C, H, W) batches — NOT individual (C, H, W) tensors.

    Transforms in __getitem__ run per-image (unavoidable — each image is a separate
    file), but the DataLoader collates individual tensors into batches automatically.
    This test proves that the training loop always sees full batches.
    """
    import torch

    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, _IMG_SIZE, _BATCH_SIZE, augment=False, use_sampler=False
    )

    for split_name, loader in [("train", train_loader), ("val", val_loader)]:
        images, labels = next(iter(loader))

        # must be 4-D: (Batch, Channels, Height, Width)
        assert images.ndim == 4, \
            f"[{split_name}] Expected 4D batch (B,C,H,W), got {images.ndim}D tensor"
        assert images.shape[1:] == (3, _IMG_SIZE, _IMG_SIZE), \
            f"[{split_name}] Wrong spatial/channel dims: {images.shape[1:]}"
        assert images.dtype == torch.float32, \
            f"[{split_name}] Expected float32, got {images.dtype}"

        # labels must be a 1-D vector, one entry per image in the batch
        assert labels.ndim == 1, \
            f"[{split_name}] Labels must be 1D, got {labels.ndim}D"
        assert labels.shape[0] == images.shape[0], \
            f"[{split_name}] Batch size mismatch: images {images.shape[0]} vs labels {labels.shape[0]}"

        print(f"[PASS] {split_name} DataLoader batch: images={tuple(images.shape)}, "
              f"labels={tuple(labels.shape)}, dtype={images.dtype}")


def test_no_train_val_overlap():
    """The train and val index sets must be completely disjoint — no data leakage."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.config import SEED, CLASSES

    df = pd.read_csv(CSV_PATH)
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    all_labels   = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices  = list(range(len(df)))

    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=all_labels
    )

    overlap = set(train_idx) & set(val_idx)
    assert len(overlap) == 0, f"Data leakage: {len(overlap)} indices appear in both train and val"
    assert len(train_idx) + len(val_idx) == len(df), "Indices don't cover the full dataset"
    print(f"[PASS] no train/val overlap: {len(train_idx)} train + {len(val_idx)} val = {len(df)} total")


def test_stratified_class_ratio():
    """Class proportions in train and val must each be within 2% of the full-set proportion."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.config import SEED, CLASSES

    df = pd.read_csv(CSV_PATH)
    label_to_idx = {c: i for i, c in enumerate(CLASSES)}
    all_labels   = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices  = list(range(len(df)))

    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=all_labels
    )

    full_labels  = [all_labels[i] for i in all_indices]
    train_labels = [all_labels[i] for i in train_idx]
    val_labels   = [all_labels[i] for i in val_idx]

    for cls_idx, cls_name in enumerate(CLASSES):
        full_pct  = full_labels.count(cls_idx)  / len(full_labels)
        train_pct = train_labels.count(cls_idx) / len(train_labels)
        val_pct   = val_labels.count(cls_idx)   / len(val_labels)

        assert abs(train_pct - full_pct) < 0.02, \
            f"{cls_name}: train proportion {train_pct:.3f} drifts > 2% from full {full_pct:.3f}"
        assert abs(val_pct - full_pct) < 0.02, \
            f"{cls_name}: val proportion {val_pct:.3f} drifts > 2% from full {full_pct:.3f}"

    print(f"[PASS] stratified split: all 9 classes within 2% proportion in train and val")


def test_gray_transforms():
    """get_gray_transforms must output (1, H, W) tensors, values roughly in [-3, 3]."""
    import torch
    ds = PokemonDataset(TRAIN_DIR, get_gray_transforms(_IMG_SIZE), csv_path=CSV_PATH)
    tensor, label = ds[0]
    assert tensor.shape == (1, _IMG_SIZE, _IMG_SIZE), \
        f"Expected (1, {_IMG_SIZE}, {_IMG_SIZE}), got {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert tensor.min() > -4.0 and tensor.max() < 4.0, \
        f"Gray values out of range: min={tensor.min():.2f}, max={tensor.max():.2f}"
    print(f"[PASS] gray transform: shape={tensor.shape}, range=[{tensor.min():.2f}, {tensor.max():.2f}]")


def test_gray_equalize_transforms():
    """get_gray_transforms(equalize=True) must output same shape as without equalization."""
    import torch
    ds_eq = PokemonDataset(TRAIN_DIR, get_gray_transforms(_IMG_SIZE, equalize=True), csv_path=CSV_PATH)
    tensor, _ = ds_eq[0]
    assert tensor.shape == (1, _IMG_SIZE, _IMG_SIZE), \
        f"Expected (1, {_IMG_SIZE}, {_IMG_SIZE}) with equalize, got {tensor.shape}"
    print(f"[PASS] gray+equalize transform: shape={tensor.shape}, range=[{tensor.min():.2f}, {tensor.max():.2f}]")


def test_gray_loaders():
    """get_train_val_loaders with grayscale=True must yield (B, 1, H, W) batches."""
    import torch
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, _IMG_SIZE, _BATCH_SIZE,
        augment=False, use_sampler=False, grayscale=True,
    )
    images, labels = next(iter(train_loader))
    assert images.shape[1] == 1, f"Expected 1 channel (gray), got {images.shape[1]}"
    assert images.shape[2:] == (_IMG_SIZE, _IMG_SIZE), f"Wrong spatial dims: {images.shape[2:]}"
    print(f"[PASS] gray loader batch: {tuple(images.shape)} (1 channel as expected)")


if __name__ == "__main__":
    print("=" * 60)
    print("dataset_test.py — running all tests")
    print("=" * 60)
    test_dataset_length()
    test_item_shape_and_dtype()
    test_label_range()
    test_class_weights()
    test_augment_same_shape()
    test_inference_mode()
    test_dataloader_batch_shapes()
    test_no_train_val_overlap()
    test_stratified_class_ratio()
    test_gray_transforms()
    test_gray_equalize_transforms()
    test_gray_loaders()
    # stratified split iterates all loaders — slowest, keep last
    test_stratified_split()
    print("=" * 60)
    print("All dataset tests passed.")
    print("=" * 60)
