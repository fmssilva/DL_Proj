# EDA visualizations — saves all figures to outputs/plots/ and returns them.
# No stdout stats here — see eda.py. Called from task1_mlp.py.

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import CLASSES, OUT_DIR, SEED


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to outputs/plots/<name>.png."""
    out = OUT_DIR / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=120)


# ── plot functions ────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """
    Horizontal bar chart of sample counts per class.
    Sorted descending, count + % annotated on bars — highlights imbalance clearly.
    """
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["class", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)
    counts = counts.sort_values("count", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(counts["class"], counts["count"], color="steelblue")

    # annotate each bar with count and percentage
    for bar, (_, row) in zip(bars, counts.iterrows()):
        ax.text(
            bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            f"{row['count']}  ({row['pct']}%)",
            va="center", fontsize=9,
        )

    ax.set_xlabel("Number of samples")
    ax.set_title("Class Distribution (train set)")
    ax.invert_yaxis()
    fig.tight_layout()

    _save(fig, "plot_class_distribution")
    return fig


def plot_sample_images(img_dir: Path, df: pd.DataFrame, n_per_class: int = 4) -> plt.Figure:
    """
    Grid of n_per_class random sample images per class.
    Fixed seed so the grid is reproducible across runs.
    """
    img_dir = Path(img_dir)
    rng     = random.Random(SEED)

    n_cols = n_per_class
    n_rows = len(CLASSES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))

    for row_idx, cls in enumerate(CLASSES):
        uuids  = df[df["label"] == cls]["Id"].tolist()
        sample = rng.sample(uuids, min(n_per_class, len(uuids)))

        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx < len(sample):
                img_path = img_dir / f"{sample[col_idx]}.png"
                if img_path.exists():
                    ax.imshow(Image.open(img_path))
            if col_idx == 0:
                ax.set_title(cls, fontsize=8, loc="left")

    fig.suptitle("Sample images per class", fontsize=12)
    fig.tight_layout()
    _save(fig, "plot_sample_images")
    return fig


def plot_average_image_per_class(img_dir: Path, df: pd.DataFrame) -> plt.Figure:
    """
    Compute mean pixel image per class and display as a 1-row grid.
    High intra-class variance shows up as blurry mean — informs augmentation choices.
    """
    img_dir = Path(img_dir)
    n_cols  = len(CLASSES)
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2, 2.5))

    for idx, cls in enumerate(CLASSES):
        uuids = df[df["label"] == cls]["Id"].tolist()
        imgs  = []

        for uid in uuids:
            p = img_dir / f"{uid}.png"
            if p.exists():
                arr = np.array(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32)
                imgs.append(arr)

        if imgs:
            mean_img = np.mean(imgs, axis=0).astype(np.uint8)
            axes[idx].imshow(mean_img)
        axes[idx].set_title(cls, fontsize=8)
        axes[idx].axis("off")

    fig.suptitle("Average image per class", fontsize=12)
    fig.tight_layout()
    _save(fig, "plot_average_image_per_class")
    return fig


def plot_pixel_statistics(img_dir: Path, df: pd.DataFrame) -> plt.Figure:
    """
    Per-channel (R/G/B) mean and std across the dataset.
    Prints computed values — tells you whether ImageNet normalisation is a good fit.
    """
    img_dir = Path(img_dir)
    # accumulate running stats — keep memory low by not loading all images at once
    channel_sum  = np.zeros(3, dtype=np.float64)
    channel_sq   = np.zeros(3, dtype=np.float64)
    n_pixels     = 0

    for uid in df["Id"]:
        p = img_dir / f"{uid}.png"
        if not p.exists():
            continue
        arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = arr.shape
        channel_sum  += arr.reshape(-1, 3).sum(axis=0)
        channel_sq   += (arr ** 2).reshape(-1, 3).sum(axis=0)
        n_pixels     += h * w

    mean = channel_sum / n_pixels
    std  = np.sqrt(channel_sq / n_pixels - mean ** 2)

    print("Per-channel pixel stats (normalised 0-1):")
    for i, ch in enumerate(["R", "G", "B"]):
        print(f"  {ch}: mean={mean[i]:.4f}, std={std[i]:.4f}")
    print(f"ImageNet reference: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    channels = ["R", "G", "B"]
    colors   = ["tomato", "limegreen", "royalblue"]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, values, title in zip(axes, [mean, std], ["Mean", "Std"]):
        ax.bar(channels, values, color=colors, alpha=0.8)
        ax.set_ylim(0, values.max() * 1.3)
        ax.set_title(f"Per-channel {title}")
        ax.set_ylabel(title)
        for i, v in enumerate(values):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    fig.suptitle("Pixel Statistics per Channel", fontsize=12)
    fig.tight_layout()
    _save(fig, "plot_pixel_statistics")
    return fig


def plot_pixel_intensity_histogram(
    img_dir: Path, df: pd.DataFrame, n_samples: int = 200
) -> plt.Figure:
    """
    R/G/B overlay histogram on a random subsample of n_samples images.
    Confirms RGB (not grayscale), helps spot blank or corrupted images.
    """
    img_dir = Path(img_dir)
    rng     = random.Random(SEED)
    sample  = rng.sample(df["Id"].tolist(), min(n_samples, len(df)))

    r_vals, g_vals, b_vals = [], [], []

    for uid in sample:
        p = img_dir / f"{uid}.png"
        if not p.exists():
            continue
        arr = np.array(Image.open(p).convert("RGB").resize((32, 32)))
        r_vals.append(arr[:, :, 0].flatten())
        g_vals.append(arr[:, :, 1].flatten())
        b_vals.append(arr[:, :, 2].flatten())

    r_all = np.concatenate(r_vals)
    g_all = np.concatenate(g_vals)
    b_all = np.concatenate(b_vals)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, 257, 4)
    ax.hist(r_all, bins=bins, color="tomato",     alpha=0.5, label="R", density=True)
    ax.hist(g_all, bins=bins, color="limegreen",  alpha=0.5, label="G", density=True)
    ax.hist(b_all, bins=bins, color="royalblue",  alpha=0.5, label="B", density=True)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Density")
    ax.set_title(f"Pixel Intensity Histogram (n={len(sample)} images)")
    ax.legend()
    fig.tight_layout()

    _save(fig, "plot_pixel_intensity_histogram")
    return fig


# ── local test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as _pd

    CSV_PATH  = Path("data/train_labels.csv")
    TRAIN_DIR = Path("data/Train")

    # use a 50-image subset so this runs quickly during dev
    df_full   = _pd.read_csv(CSV_PATH)
    df_subset = df_full.sample(n=50, random_state=SEED).reset_index(drop=True)

    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    fig1 = plot_class_distribution(df_subset)
    assert fig1 is not None, "plot_class_distribution returned None"
    print("[PASS] plot_class_distribution")

    fig2 = plot_sample_images(TRAIN_DIR, df_full, n_per_class=2)
    assert fig2 is not None, "plot_sample_images returned None"
    print("[PASS] plot_sample_images")

    fig3 = plot_average_image_per_class(TRAIN_DIR, df_subset)
    assert fig3 is not None, "plot_average_image_per_class returned None"
    print("[PASS] plot_average_image_per_class")

    fig4 = plot_pixel_statistics(TRAIN_DIR, df_subset)
    assert fig4 is not None, "plot_pixel_statistics returned None"
    print("[PASS] plot_pixel_statistics")

    fig5 = plot_pixel_intensity_histogram(TRAIN_DIR, df_subset, n_samples=50)
    assert fig5 is not None, "plot_pixel_intensity_histogram returned None"
    print("[PASS] plot_pixel_intensity_histogram")

    print("All eda_plots tests passed (50-image subset).")
    plt.close("all")
