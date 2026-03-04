# Training history and confusion matrix plots. Called by all 3 task entry-points.

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Save to given path (caller decides the name), create dirs if needed."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)


def plot_history(history: dict, out_path: Path, title: str = "") -> plt.Figure:
    """
    Plot train/val loss and macro_f1 curves over epochs on the same axes pair.
    history must have keys: train_loss, val_loss, train_f1, val_f1 (lists of floats).
    title: optional prefix shown in each subplot title — useful when comparing experiments.
    NaN values in train_f1 (e.g. MLP — not tracked per epoch) are simply not plotted.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    prefix = f"{title} — " if title else ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # loss subplot
    ax1.plot(epochs, history["train_loss"], label="train_loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"],   label="val_loss",   linewidth=1.5, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{prefix}Loss")
    ax1.legend()

    # macro f1 subplot — skip train_f1 if all NaN
    train_f1 = history.get("train_f1", [])
    if train_f1 and not all(math.isnan(v) for v in train_f1):
        ax2.plot(epochs, train_f1, label="train_f1", linewidth=1.5)
    ax2.plot(epochs, history["val_f1"], label="val_f1", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title(f"{prefix}Macro F1")
    ax2.legend()

    fig.tight_layout()
    _save(fig, out_path)
    return fig


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    classes: list,
    out_path: Path,
) -> plt.Figure:
    """
    Seaborn heatmap of the confusion matrix, normalised by row (recall per class).
    Rows = true class, columns = predicted class.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))), normalize="true")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    _save(fig, out_path)
    return fig


# ── sanity checks ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.config import CLASSES, OUT_DIR

    (OUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

    # dummy history
    dummy_history = {
        "train_loss": [2.2, 2.0, 1.8, 1.6, 1.5],
        "val_loss":   [2.3, 2.1, 1.9, 1.8, 1.7],
        "train_f1":   [0.1, 0.2, 0.3, 0.4, 0.45],
        "val_f1":     [0.08, 0.18, 0.28, 0.35, 0.40],
    }
    history_path = OUT_DIR / "plots" / "test_history.png"
    fig1 = plot_history(dummy_history, history_path)
    assert history_path.exists(), "plot_history did not save file"
    assert fig1 is not None
    print(f"[PASS] plot_history saved to {history_path}")
    plt.close(fig1)

    # dummy confusion matrix (50 samples, 9 classes)
    rng     = np.random.default_rng(0)
    y_true  = rng.integers(0, 9, size=50).tolist()
    y_pred  = rng.integers(0, 9, size=50).tolist()
    cm_path = OUT_DIR / "plots" / "test_confusion_matrix.png"
    fig2    = plot_confusion_matrix(y_true, y_pred, CLASSES, cm_path)
    assert cm_path.exists(), "plot_confusion_matrix did not save file"
    assert fig2 is not None
    print(f"[PASS] plot_confusion_matrix saved to {cm_path}")
    plt.close(fig2)

    print("All evaluation plots tests passed.")
