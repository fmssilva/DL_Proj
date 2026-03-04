# Local tests for dataset.py — run on CPU before touching Colab.
# Run with: python src/datasets/dataset_test.py  (from assignment_1/ root)

from collections import Counter
from pathlib import Path

from src.config import BATCH_SIZE, CLASSES, IMG_SIZE_SMALL, NUM_CLASSES
from src.datasets.dataset import (
    PokemonDataset,
    compute_class_weights,
    get_augment_transforms,
    get_base_transforms,
    get_train_val_loaders,
)

# ── paths (relative to assignment_1/) ─────────────────────────────────────────
CSV_PATH  = Path("data/train_labels.csv")
TRAIN_DIR = Path("data/Train")
TEST_DIR  = Path("data/Test")


def test_dataset_length():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(IMG_SIZE_SMALL), csv_path=CSV_PATH)
    assert len(ds) == 3600, f"Expected 3600 samples, got {len(ds)}"
    print(f"[PASS] dataset length: {len(ds)}")


def test_item_shape_and_dtype():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(IMG_SIZE_SMALL), csv_path=CSV_PATH)
    tensor, label = ds[0]

    # shape must be (C, H, W) after transform
    assert tensor.shape == (3, IMG_SIZE_SMALL, IMG_SIZE_SMALL), \
        f"Expected (3,{IMG_SIZE_SMALL},{IMG_SIZE_SMALL}), got {tensor.shape}"
    assert tensor.dtype == __import__("torch").float32, f"Expected float32, got {tensor.dtype}"

    # after ImageNet normalisation values live roughly in [-3, 3]
    assert tensor.min() > -4.0 and tensor.max() < 4.0, \
        f"Values out of expected range: min={tensor.min():.2f}, max={tensor.max():.2f}"

    print(f"[PASS] item shape {tensor.shape}, dtype {tensor.dtype}, "
          f"range [{tensor.min():.2f}, {tensor.max():.2f}]")


def test_label_range():
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(IMG_SIZE_SMALL), csv_path=CSV_PATH)
    labels = [ds[i][1] for i in range(len(ds))]
    assert min(labels) == 0 and max(labels) == NUM_CLASSES - 1, \
        f"Labels must be in [0, {NUM_CLASSES-1}], got [{min(labels)}, {max(labels)}]"
    print(f"[PASS] labels in [0, {NUM_CLASSES-1}], unique: {sorted(set(labels))}")


def test_stratified_split():
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE_SMALL, BATCH_SIZE, augment=False, use_sampler=False
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
    from src.config import CLASSES as CLS
    df = pd.read_csv(CSV_PATH)
    label_to_idx = {c: i for i, c in enumerate(CLS)}
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
    base_ds  = PokemonDataset(TRAIN_DIR, get_base_transforms(IMG_SIZE_SMALL),   csv_path=CSV_PATH)
    aug_ds   = PokemonDataset(TRAIN_DIR, get_augment_transforms(IMG_SIZE_SMALL), csv_path=CSV_PATH)
    t_base,  _ = base_ds[0]
    t_aug,   _ = aug_ds[0]
    assert t_base.shape == t_aug.shape, \
        f"Augmented shape {t_aug.shape} != base shape {t_base.shape}"
    print(f"[PASS] augmented output shape matches base: {t_aug.shape}")


def test_inference_mode():
    ds = PokemonDataset(TEST_DIR, get_base_transforms(IMG_SIZE_SMALL), csv_path=None)
    assert len(ds) == 900, f"Expected 900 test images, got {len(ds)}"

    tensor, uuid = ds[0]
    assert tensor.shape == (3, IMG_SIZE_SMALL, IMG_SIZE_SMALL), \
        f"Test tensor shape wrong: {tensor.shape}"
    assert isinstance(uuid, str) and len(uuid) > 0, f"Expected UUID string, got: {uuid!r}"

    print(f"[PASS] inference mode: {len(ds)} images, sample uuid={uuid!r}, shape={tensor.shape}")


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
    # stratified split iterates all loaders — slowest, keep last
    test_stratified_split()
    print("=" * 60)
    print("All dataset tests passed.")
    print("=" * 60)
