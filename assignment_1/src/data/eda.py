# EDA stats: class counts, image sizes, and data integrity check.
# No plots here — see eda_plots.py. Called from task1_mlp.py.

from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.config import CLASSES


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count samples per class, compute percentages, and print imbalance summary.
    Returns a DataFrame sorted by count descending.
    """
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)
    counts = counts.sort_values("count", ascending=False).reset_index(drop=True)

    imbalance_ratio = counts["count"].max() / counts["count"].min()

    print("Class distribution:")
    for _, row in counts.iterrows():
        print(f"  {row['class']:12s}: {row['count']:4d}  ({row['pct']:.1f}%)")
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}x")

    return counts


def image_size_distribution(img_dir: Path) -> dict:
    """
    Check all image sizes in img_dir. Logs a warning if sizes aren't uniform.
    Returns a dict of {(W,H): count} — useful to confirm resize is safe.
    """
    img_dir  = Path(img_dir)
    size_map = {}

    for p in img_dir.glob("*.png"):
        try:
            w, h = Image.open(p).size
            key  = (w, h)
            size_map[key] = size_map.get(key, 0) + 1
        except Exception as e:
            print(f"Could not read {p.name}: {e}")

    if len(size_map) == 1:
        size, cnt = list(size_map.items())[0]
        print(f"Image sizes: all {cnt} images are {size[0]}x{size[1]}")
    else:
        print(f"WARNING: {len(size_map)} different image sizes found:")
        for size, cnt in sorted(size_map.items(), key=lambda x: -x[1]):
            print(f"  {size[0]}x{size[1]}: {cnt} images")

    return size_map


def check_data_integrity(img_dir: Path, df: Optional[pd.DataFrame] = None) -> tuple:
    """
    Try to open every image in img_dir. Returns (valid_count, invalid_count).
    Logs the filename of any unreadable image — catch corruptions before training.
    df is optional; if given, also checks that every CSV entry has a matching file.
    """
    img_dir = Path(img_dir)
    valid   = 0
    invalid = 0

    for p in img_dir.glob("*.png"):
        try:
            img = Image.open(p)
            img.verify()   # verify without fully loading into memory
            valid += 1
        except (UnidentifiedImageError, Exception) as e:
            print(f"Unreadable image: {p.name} — {e}")
            invalid += 1

    print(f"Integrity check: {valid} valid, {invalid} invalid images in {img_dir}")

    if df is not None:
        # check every CSV UUID has a matching file on disk
        missing = [uid for uid in df["Id"] if not (img_dir / f"{uid}.png").exists()]
        if missing:
            print(f"WARNING: {len(missing)} CSV entries have no matching image file")
        else:
            print(f"All {len(df)} CSV entries have matching image files")

    return valid, invalid


# ── local runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH  = Path("data/train_labels.csv")
    TRAIN_DIR = Path("data/Train")
    TEST_DIR  = Path("data/Test")

    df = pd.read_csv(CSV_PATH)

    print("\n--- class_distribution ---")
    class_distribution(df)

    print("\n--- image_size_distribution (Train) ---")
    image_size_distribution(TRAIN_DIR)

    print("\n--- check_data_integrity (Train) ---")
    valid, invalid = check_data_integrity(TRAIN_DIR, df)
    assert invalid == 0, f"{invalid} invalid images found in Train/"

    print("\n--- check_data_integrity (Test, no CSV) ---")
    check_data_integrity(TEST_DIR)

    print("\nAll EDA stats completed successfully.")
