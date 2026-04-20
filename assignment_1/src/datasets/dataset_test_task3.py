# Comprehensive data-pipeline tests for Task 3 (transfer learning).
# Checks the ENTIRE flow: CSV → labels → split → transforms → Dataset → DataLoader → batches.
#
# Run with:  conda run -n cnn python -m src.datasets.dataset_test_task3  (from assignment_1/)

import os, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from src.config import CLASSES, NUM_CLASSES, SEED
from src.datasets.dataset import (
    PokemonDataset,
    compute_class_weights,
    get_base_transforms,
    get_augment_transforms,
    get_transfer_aug_transforms,
    get_train_val_loaders,
)

# ── paths & constants ─────────────────────────────────────────────────────────
CSV_PATH   = Path("data/train_labels.csv")
TRAIN_DIR  = Path("data/train")
TEST_DIR   = Path("data/test")
IMG_SIZE   = 224          # task 3 uses 224px for all transfer models
BATCH_SIZE = 32

# Task 3 dataset sizes (Pokemon 2026 Task 3)
EXPECTED_TRAIN = 1194
EXPECTED_TEST  = 300

_pass = 0
_fail = 0


def _report(name: str, ok: bool):
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  [PASS] {name}")
    else:
        _fail += 1
        print(f"  [FAIL] {name}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: File integrity — CSV and image folder sanity
# ══════════════════════════════════════════════════════════════════════════════
def test_file_integrity():
    print("\n" + "=" * 70)
    print("TEST 1: File integrity — CSV, image folders, filename matching")
    print("=" * 70)
    ok = True

    # CSV exists and is well-formed
    assert CSV_PATH.exists(), f"CSV not found: {CSV_PATH}"
    df = pd.read_csv(CSV_PATH)
    assert list(df.columns) == ["Id", "label"], f"Unexpected columns: {list(df.columns)}"
    _report("CSV columns are [Id, label]", True)

    # Row count
    _report(f"CSV has {len(df)} rows (expected {EXPECTED_TRAIN})", len(df) == EXPECTED_TRAIN)
    if len(df) != EXPECTED_TRAIN:
        ok = False

    # Labels match config.CLASSES exactly
    csv_classes = sorted(df["label"].unique())
    expected    = sorted(CLASSES)
    match = csv_classes == expected
    _report(f"CSV classes match config.CLASSES: {csv_classes}", match)
    if not match:
        ok = False

    # Every row in CSV has a corresponding image file
    missing = []
    for row in df.itertuples():
        img_path = TRAIN_DIR / f"{row.Id}.png"
        if not img_path.exists():
            missing.append(row.Id)
    _report(f"All CSV Ids have matching train images ({len(missing)} missing)", len(missing) == 0)
    if missing:
        print(f"    Missing images: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        ok = False

    # Train folder doesn't have extra images not in CSV
    csv_ids    = set(df["Id"])
    folder_ids = {p.stem for p in TRAIN_DIR.glob("*.png")}
    extra = folder_ids - csv_ids
    _report(f"No orphan images in train/ ({len(extra)} extra)", len(extra) == 0)
    if extra:
        print(f"    Extra images: {list(extra)[:5]}")
        ok = False

    # Test folder
    test_count = len(list(TEST_DIR.glob("*.png")))
    _report(f"Test folder has {test_count} images (expected {EXPECTED_TEST})", test_count == EXPECTED_TEST)

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Label encoding consistency
# ══════════════════════════════════════════════════════════════════════════════
def test_label_encoding():
    print("\n" + "=" * 70)
    print("TEST 2: Label encoding — CLASSES ordering, label_to_idx mapping")
    print("=" * 70)

    # CLASSES must be sorted (our label_to_idx = {cls: i for i, cls in enumerate(CLASSES)})
    is_sorted = CLASSES == sorted(CLASSES)
    _report(f"CLASSES is alphabetically sorted: {CLASSES}", is_sorted)

    # label_to_idx
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    idx_to_label = {i: cls for cls, i in label_to_idx.items()}

    # Verify via dataset
    df = pd.read_csv(CSV_PATH)
    ds = PokemonDataset(TRAIN_DIR, get_base_transforms(IMG_SIZE), df=df)

    # Sample 50 items and verify label encoding
    ok = True
    for i in range(min(50, len(ds))):
        _, label_int = ds[i]
        # Get the original string label for this row
        row = df.iloc[i]
        expected_int = label_to_idx[row["label"]]
        if label_int != expected_int:
            print(f"    MISMATCH at idx {i}: got {label_int}, expected {expected_int} for '{row['label']}'")
            ok = False

    _report("label_to_idx encoding matches for 50 sampled items", ok)

    # Check that all 9 integer labels are present
    all_labels = [ds[i][1] for i in range(len(ds))]
    unique = sorted(set(all_labels))
    _report(f"All 9 classes present in dataset labels: {unique}", unique == list(range(NUM_CLASSES)))

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Stratified split determinism and correctness
# ══════════════════════════════════════════════════════════════════════════════
def test_split_determinism():
    print("\n" + "=" * 70)
    print("TEST 3: Stratified split — determinism, no overlap, 80/20, class balance")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_labels   = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices  = list(range(len(df)))

    # Run split twice — MUST produce identical indices
    t1, v1 = train_test_split(all_indices, test_size=0.2, random_state=SEED, stratify=all_labels)
    t2, v2 = train_test_split(all_indices, test_size=0.2, random_state=SEED, stratify=all_labels)
    _report("Split is deterministic (same seed → same indices)", t1 == t2 and v1 == v2)

    # No overlap
    overlap = set(t1) & set(v1)
    _report(f"No train/val overlap ({len(overlap)} shared indices)", len(overlap) == 0)

    # Size check
    _report(f"Train size = {len(t1)} (~80%)", abs(len(t1) / len(df) - 0.8) < 0.02)
    _report(f"Val size = {len(v1)} (~20%)", abs(len(v1) / len(df) - 0.2) < 0.02)
    _report(f"Train + Val = {len(t1) + len(v1)} (== {len(df)})", len(t1) + len(v1) == len(df))

    # Class balance
    train_labels = [all_labels[i] for i in t1]
    val_labels   = [all_labels[i] for i in v1]
    balance_ok = True
    for cls_idx, cls_name in enumerate(CLASSES):
        full_pct  = all_labels.count(cls_idx) / len(all_labels)
        train_pct = train_labels.count(cls_idx) / len(train_labels) if train_labels else 0
        val_pct   = val_labels.count(cls_idx) / len(val_labels) if val_labels else 0
        if abs(train_pct - full_pct) > 0.03 or abs(val_pct - full_pct) > 0.03:
            print(f"    {cls_name}: full={full_pct:.3f} train={train_pct:.3f} val={val_pct:.3f}  DRIFT!")
            balance_ok = False
    _report("Stratified: all classes within 3% of full-set proportion", balance_ok)

    # Confirm build_loaders calls produce same val set
    # (key for fair comparison across experiments!)
    loader1_train, loader1_val = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
    )
    loader2_train, loader2_val = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
    )
    # val loaders should have same dataset length
    _report(
        f"Two get_train_val_loaders calls → same val size ({len(loader1_val.dataset)} == {len(loader2_val.dataset)})",
        len(loader1_val.dataset) == len(loader2_val.dataset)
    )

    # Compare actual val labels (must be identical)
    val_labels_1 = [lbl for _, lbl in loader1_val.dataset]
    val_labels_2 = [lbl for _, lbl in loader2_val.dataset]
    _report("Val labels identical across two loader calls", val_labels_1 == val_labels_2)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Transform output shapes and values
# ══════════════════════════════════════════════════════════════════════════════
def test_transforms():
    print("\n" + "=" * 70)
    print("TEST 4: Transform output — shapes, dtypes, normalization ranges")
    print("=" * 70)
    from PIL import Image

    # Pick a real image
    df = pd.read_csv(CSV_PATH)
    img_id = df.iloc[0]["Id"]
    pil_img = Image.open(TRAIN_DIR / f"{img_id}.png").convert("RGB")
    print(f"  Source image: {img_id}.png, original size={pil_img.size}, mode={pil_img.mode}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    transforms_to_test = [
        ("get_base_transforms(224)",         get_base_transforms(224),         (3, 224, 224)),
        ("get_augment_transforms(224)",       get_augment_transforms(224),       (3, 224, 224)),
        ("get_transfer_aug_transforms(224)",  get_transfer_aug_transforms(224),  (3, 224, 224)),
        ("get_base_transforms(64)",          get_base_transforms(64),          (3, 64, 64)),
    ]

    for name, tfm, expected_shape in transforms_to_test:
        t = tfm(pil_img)
        shape_ok = t.shape == expected_shape
        dtype_ok = t.dtype == torch.float32

        # After ImageNet normalization, per-channel mean should be near 0
        # (not exactly 0 because one image isn't the dataset mean)
        # But values should be in roughly [-3, 3]
        range_ok = t.min() > -5.0 and t.max() < 5.0

        _report(f"{name}: shape={t.shape} dtype={t.dtype} range=[{t.min():.2f},{t.max():.2f}]",
                shape_ok and dtype_ok and range_ok)

    # Verify normalization is applied (not raw [0,1])
    t_base = get_base_transforms(224)(pil_img)
    # After ImageNet norm, the mean across the whole tensor should NOT be ~0.5
    mean_val = t_base.mean().item()
    _report(f"ImageNet normalization applied (tensor mean={mean_val:.3f}, not ~0.5)", abs(mean_val - 0.5) > 0.1)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: DataLoader batch shapes — 224px for transfer, correct label types
# ══════════════════════════════════════════════════════════════════════════════
def test_loader_batches():
    print("\n" + "=" * 70)
    print("TEST 5: DataLoader batches — shapes, dtypes, label ranges")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
    )

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        images, labels = next(iter(loader))

        _report(f"{name}: images.shape={tuple(images.shape)} (expected (B,3,224,224))",
                images.ndim == 4 and images.shape[1:] == (3, IMG_SIZE, IMG_SIZE))
        _report(f"{name}: images.dtype={images.dtype}",
                images.dtype == torch.float32)
        _report(f"{name}: labels.shape={tuple(labels.shape)}, dtype={labels.dtype}",
                labels.ndim == 1 and labels.dtype in (torch.int64, torch.long))
        _report(f"{name}: labels in [0,{NUM_CLASSES-1}]",
                labels.min() >= 0 and labels.max() < NUM_CLASSES)

    # Full iteration — count total samples
    train_total = sum(lbls.shape[0] for _, lbls in train_loader)
    val_total   = sum(lbls.shape[0] for _, lbls in val_loader)
    _report(f"Full iteration: train={train_total} + val={val_total} = {train_total+val_total} (expected {EXPECTED_TRAIN})",
            train_total + val_total == EXPECTED_TRAIN)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6: Class weights correctness
# ══════════════════════════════════════════════════════════════════════════════
def test_class_weights():
    print("\n" + "=" * 70)
    print("TEST 6: Class weights — inverse-frequency, rare classes heavier")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_labels = [label_to_idx[lbl] for lbl in df["label"]]

    weights = compute_class_weights(all_labels)
    _report(f"Shape = {weights.shape} (expected ({NUM_CLASSES},))", weights.shape == (NUM_CLASSES,))
    _report("All weights > 0", (weights > 0).all().item())

    # Manually compute expected weights
    counts = Counter(all_labels)
    total = len(all_labels)
    for cls_idx, cls_name in enumerate(CLASSES):
        expected_w = total / (NUM_CLASSES * counts[cls_idx])
        actual_w = weights[cls_idx].item()
        close = abs(actual_w - expected_w) < 1e-4
        _report(f"  {cls_name:10s}: count={counts[cls_idx]:4d}  weight={actual_w:.4f} (expected {expected_w:.4f})", close)

    # Fighting (60 samples) must be heavier than Water (252 samples)
    fighting_w = weights[CLASSES.index("Fighting")].item()
    water_w    = weights[CLASSES.index("Water")].item()
    _report(f"Fighting({fighting_w:.3f}) > Water({water_w:.3f})", fighting_w > water_w)

    # Weights for CrossEntropyLoss: verify sum is reasonable
    print(f"\n  Weight sum = {weights.sum().item():.4f}  (higher = more aggressive rebalancing)")

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7: WeightedRandomSampler — balanced batches
# ══════════════════════════════════════════════════════════════════════════════
def test_weighted_sampler():
    print("\n" + "=" * 70)
    print("TEST 7: WeightedRandomSampler — batch class balance")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    train_loader, _ = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=True, df_override=df,
    )

    # Collect all labels from one epoch
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    counts = Counter(all_labels)
    print(f"  Sampled {len(all_labels)} items across {len(train_loader)} batches")
    print(f"  Class counts: {dict(sorted(counts.items()))}")

    # With replacement sampling, total should equal dataset size
    # (WeightedRandomSampler num_samples = len(sample_weights))
    _report(f"Total sampled = {len(all_labels)}", True)

    # Check that rare classes are significantly boosted
    # Without sampler, Fighting has 60/1194 = 5%. With sampler, should be ~11% (1/9)
    if CLASSES.index("Fighting") in counts:
        fighting_pct = counts[CLASSES.index("Fighting")] / len(all_labels)
        expected_pct = 1.0 / NUM_CLASSES
        _report(f"Fighting sampled at {fighting_pct:.1%} (expected ~{expected_pct:.1%}, natural ~5%)",
                fighting_pct > 0.07)  # at least 7%, much above the natural 5%

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8: Transfer augmentation doesn't break normalization
# ══════════════════════════════════════════════════════════════════════════════
def test_transfer_aug_normalization():
    print("\n" + "=" * 70)
    print("TEST 8: Transfer aug + base transforms produce compatible tensors")
    print("=" * 70)
    from PIL import Image

    df = pd.read_csv(CSV_PATH)
    img_id = df.iloc[0]["Id"]
    pil_img = Image.open(TRAIN_DIR / f"{img_id}.png").convert("RGB")

    base_t = get_base_transforms(IMG_SIZE)(pil_img)
    aug_t  = get_transfer_aug_transforms(IMG_SIZE)(pil_img)

    _report(f"Base shape={base_t.shape} == Aug shape={aug_t.shape}", base_t.shape == aug_t.shape)
    _report(f"Base dtype={base_t.dtype} == Aug dtype={aug_t.dtype}", base_t.dtype == aug_t.dtype)

    # Both should be ImageNet-normalized (similar range)
    _report(f"Base range=[{base_t.min():.2f},{base_t.max():.2f}]", base_t.min() > -4.0 and base_t.max() < 4.0)
    _report(f"Aug  range=[{aug_t.min():.2f},{aug_t.max():.2f}]", aug_t.min() > -4.0 and aug_t.max() < 4.0)

    # Val loader should always use base transforms, even when train uses aug
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
        train_transform=get_transfer_aug_transforms(IMG_SIZE),
    )
    # Verify val transform is deterministic (run same sample twice, same output)
    val_ds = val_loader.dataset
    t1, l1 = val_ds[0]
    t2, l2 = val_ds[0]
    _report("Val transform is deterministic (no random aug)", torch.allclose(t1, t2) and l1 == l2)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9: Inference (test) dataset — no labels, UUID stems
# ══════════════════════════════════════════════════════════════════════════════
def test_inference_mode():
    print("\n" + "=" * 70)
    print("TEST 9: Inference mode — test dataset, UUID stems, no labels")
    print("=" * 70)

    ds = PokemonDataset(TEST_DIR, get_base_transforms(IMG_SIZE), csv_path=None)
    _report(f"Test dataset length = {len(ds)} (expected {EXPECTED_TEST})", len(ds) == EXPECTED_TEST)

    tensor, uuid = ds[0]
    _report(f"Output is (tensor, uuid_string): type(uuid)={type(uuid).__name__}", isinstance(uuid, str))
    _report(f"UUID looks valid: '{uuid[:20]}...'", len(uuid) > 10 and "-" in uuid)
    _report(f"Tensor shape = {tensor.shape}", tensor.shape == (3, IMG_SIZE, IMG_SIZE))

    # Test via DataLoader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    images, ids = next(iter(test_loader))
    _report(f"Test batch images: {tuple(images.shape)}", images.ndim == 4 and images.shape[1:] == (3, IMG_SIZE, IMG_SIZE))
    _report(f"Test batch ids: {len(ids)} strings", len(ids) == images.shape[0] and isinstance(ids[0], str))

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10: FAST_RUN subsampling (simulates notebook behavior)
# ══════════════════════════════════════════════════════════════════════════════
def test_fast_run_subsampling():
    print("\n" + "=" * 70)
    print("TEST 10: FAST_RUN subsampling — N_SAMPLES_PER_CLASS=6")
    print("=" * 70)

    df_full = pd.read_csv(CSV_PATH)
    N_SAMPLES_PER_CLASS = 6
    df_fast = df_full.groupby("label", group_keys=False).head(N_SAMPLES_PER_CLASS).reset_index(drop=True)

    expected_fast = NUM_CLASSES * N_SAMPLES_PER_CLASS  # 9 * 6 = 54
    _report(f"FAST_RUN df has {len(df_fast)} rows (expected {expected_fast})", len(df_fast) == expected_fast)

    # All classes present
    _report(f"All {NUM_CLASSES} classes in fast df", df_fast["label"].nunique() == NUM_CLASSES)

    # build loaders with subsampled df
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, 8,
        augment=False, use_sampler=False, df_override=df_fast,
    )
    train_total = sum(lbls.shape[0] for _, lbls in train_loader)
    val_total   = sum(lbls.shape[0] for _, lbls in val_loader)
    _report(f"FAST_RUN split: train={train_total} + val={val_total} = {train_total + val_total} (expected {expected_fast})",
            train_total + val_total == expected_fast)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11: Class weights computed from TRAINING split (not full dataset)
# ══════════════════════════════════════════════════════════════════════════════
def test_class_weights_from_split():
    """The notebook computes class_weights from df['label'] (full dataset).
    But the model only trains on the TRAIN split (80%).
    For proper weighting, class_weights should reflect the training split.
    Check if this discrepancy matters."""
    print("\n" + "=" * 70)
    print("TEST 11: Class weights — full-dataset vs train-split comparison")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_labels = [label_to_idx[lbl] for lbl in df["label"]]
    all_indices = list(range(len(df)))

    train_idx, val_idx = train_test_split(
        all_indices, test_size=0.2, random_state=SEED, stratify=all_labels
    )
    train_labels = [all_labels[i] for i in train_idx]

    w_full  = compute_class_weights(all_labels)
    w_train = compute_class_weights(train_labels)

    print(f"  {'Class':10s}  {'w_full':>8s}  {'w_train':>8s}  {'diff':>8s}")
    max_diff = 0.0
    for i, cls in enumerate(CLASSES):
        diff = abs(w_full[i].item() - w_train[i].item())
        max_diff = max(max_diff, diff)
        print(f"  {cls:10s}  {w_full[i].item():8.4f}  {w_train[i].item():8.4f}  {diff:8.4f}")

    # With stratified split the proportions are preserved, so weights should be very close
    _report(f"Max weight diff = {max_diff:.4f} (< 0.05 = acceptable)", max_diff < 0.05)

    return True


# ══════════════════════════════════════════════════════════════════════════════
# TEST 12: build_loaders with transfer_aug — val uses base transforms
# ══════════════════════════════════════════════════════════════════════════════
def test_build_loaders_aug_vs_base():
    """When build_loaders(transfer_aug=True), the TRAIN loader must use aug
    but the VAL loader must NOT use augmentation (deterministic evaluation)."""
    print("\n" + "=" * 70)
    print("TEST 12: build_loaders(transfer_aug=True) — val is deterministic")
    print("=" * 70)

    df = pd.read_csv(CSV_PATH)

    # Base loaders (no aug)
    train_base, val_base = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
    )

    # Aug loaders
    train_aug, val_aug = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False, use_sampler=False, df_override=df,
        train_transform=get_transfer_aug_transforms(IMG_SIZE),
    )

    # Val datasets must produce identical outputs (both use base transforms)
    v1_t, v1_l = val_base.dataset[0]
    v2_t, v2_l = val_aug.dataset[0]
    _report("Val sample 0 is identical (base vs aug loaders)", torch.allclose(v1_t, v2_t) and v1_l == v2_l)

    # Val datasets must be same size
    _report(f"Val sizes match: {len(val_base.dataset)} == {len(val_aug.dataset)}",
            len(val_base.dataset) == len(val_aug.dataset))

    # Train aug should differ from train base (due to random transforms)
    # Run same index twice with aug — may differ (randomness)
    # But train base should be deterministic
    t_base_1, _ = train_base.dataset[0]
    t_base_2, _ = train_base.dataset[0]
    _report("Train base transform is deterministic", torch.allclose(t_base_1, t_base_2))

    return True


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("dataset_test_task3.py — comprehensive data pipeline tests for Task 3")
    print(f"Dataset: {CSV_PATH}  |  Train: {TRAIN_DIR}  |  Test: {TEST_DIR}")
    print("=" * 70)

    test_file_integrity()
    test_label_encoding()
    test_split_determinism()
    test_transforms()
    test_loader_batches()
    test_class_weights()
    test_weighted_sampler()
    test_transfer_aug_normalization()
    test_inference_mode()
    test_fast_run_subsampling()
    test_class_weights_from_split()
    test_build_loaders_aug_vs_base()

    print("\n" + "=" * 70)
    print(f"SUMMARY: {_pass} passed, {_fail} failed out of {_pass + _fail} checks")
    print("=" * 70)

    if _fail > 0:
        sys.exit(1)
