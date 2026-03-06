# Post-training analysis: leaderboard bar chart and per-class F1 heatmap.
# Used in Part 3 of each task notebook. Import these and call with your
# results_tracker dict and a val_loader — no training happens here.

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import DataLoader

from ..config import CLASSES


def plot_leaderboard(
    results_tracker: dict,
    out_path: Path,
    baseline_f1: float = 0.111,
) -> plt.Figure:
    """
    Bar chart of val_macro_f1 per experiment, sorted descending.
    Best bar is highlighted orange; a dashed red line marks the random baseline.
    results_tracker: {exp_name: {"val_macro_f1": float, "train_time_s": float, ...}}
    """
    # sort by f1 descending for the chart
    items = sorted(results_tracker.items(), key=lambda x: x[1]["val_macro_f1"], reverse=True)
    names = [k for k, _ in items]
    f1s   = [v["val_macro_f1"] for _, v in items]
    times = [v["train_time_s"] for _, v in items]

    # two plots stacked vertically so many experiment names don't get cramped
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(names) * 0.7 + 2), 10))

    # F1 bar — best bar in orange, rest in steelblue
    best_f1  = max(f1s)
    colors_f1 = ["darkorange" if f == best_f1 else "steelblue" for f in f1s]
    bars0 = axes[0].bar(names, f1s, color=colors_f1)
    axes[0].axhline(baseline_f1, color="red", linestyle="--", linewidth=1,
                    label=f"random baseline ({baseline_f1:.3f})")
    axes[0].set_ylabel("Val Macro F1")
    axes[0].set_title("Experiment Leaderboard — Val Macro F1")
    axes[0].set_ylim(0, best_f1 * 1.35)
    axes[0].legend(fontsize=8)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    y_range0 = best_f1 * 1.35
    for bar, v in zip(bars0, f1s):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + y_range0 * 0.02,
            f"{v:.4f}", ha="center", va="bottom", fontsize=7,
        )

    # training time bar
    bars1 = axes[1].bar(names, times, color="slategray")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Training Time per Experiment")
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    max_t = max(times) if max(times) > 0 else 1.0
    axes[1].set_ylim(0, max_t * 1.35)
    for bar, v in zip(bars1, times):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_t * 1.35 * 0.02,
            f"{v:.0f}s", ha="center", va="bottom", fontsize=7,
        )

    fig.tight_layout()
    _save(fig, out_path)
    return fig


def compute_per_class_f1(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    classes: list = CLASSES,
) -> dict:
    """
    Run model on val_loader, return {class_name: f1_score} dict.
    Uses zero_division=0 so unseen classes score 0 rather than raising.
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # per-class F1 — returns array in the same order as `labels` arg
    per_class = f1_score(
        all_labels, all_preds,
        labels=list(range(len(classes))),
        average=None,
        zero_division=0,
    )
    return {cls: round(float(per_class[i]), 4) for i, cls in enumerate(classes)}


def plot_per_class_f1_heatmap(
    checkpoint_dir: Path,
    model_registry: dict,
    loader_fn: Callable,
    device: torch.device,
    out_path: Path,
    classes: list = CLASSES,
    highlight_best: bool = True,
    loader_fn_registry: dict = None,
) -> plt.Figure:
    """
    For every .pth in checkpoint_dir that appears in model_registry, build a model,
    load the checkpoint, compute per-class F1 on a fresh val loader, and plot a
    seaborn heatmap (rows = experiments, columns = classes).

    model_registry: {checkpoint_stem: callable() -> nn.Module}
        e.g. {"C_ls01_drop03": lambda: MLP(img_size=64, dropout=0.3)}
        Stems NOT in the registry are skipped silently (e.g. ensemble checkpoints).

    loader_fn: callable() -> (train_loader, val_loader)
        Default loader for all stems. Called fresh per model.

    loader_fn_registry: optional {stem: callable() -> (train_loader, val_loader)}
        Per-stem loader overrides. Use this for stems that need a different loader
        (e.g. grayscale experiments that need build_loaders(grayscale=True)).
        Stems not listed here fall back to loader_fn.

    highlight_best: if True, add a column "macro_avg" and sort rows by it descending.

    The heatmap colour scale is per-column (per class) — so you can instantly see
    which experiments are above/below the class average, not just the raw score.
    """
    ckpt_dir = Path(checkpoint_dir)
    _loader_overrides = loader_fn_registry or {}

    rows   = {}   # exp_name -> {class: f1}
    for ckpt_path in sorted(ckpt_dir.glob("*.pth")):
        stem = ckpt_path.stem
        if stem not in model_registry:
            continue  # skip experiments not registered (extension models, ensembles)
        model = model_registry[stem]().to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        # use per-stem loader if provided, otherwise fall back to the default loader_fn
        stem_loader_fn = _loader_overrides.get(stem, loader_fn)
        _, val_loader = stem_loader_fn()
        rows[stem] = compute_per_class_f1(model, val_loader, device, classes)

    if not rows:
        print("plot_per_class_f1_heatmap: no matching checkpoints found in registry — skipping.")
        return None

    # build DataFrame: rows = experiments, columns = classes
    import pandas as pd
    df = pd.DataFrame(rows).T[classes]   # keep class column order consistent

    if highlight_best:
        df["macro_avg"] = df.mean(axis=1)
        df = df.sort_values("macro_avg", ascending=False)

    # find the best F1 per class across all models — used to annotate the plot
    best_row = df[classes].max(axis=0)

    fig, ax = plt.subplots(figsize=(max(10, len(classes) + 2), max(6, len(df) * 0.55 + 1)))
    sns.heatmap(
        df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.4,
        cbar_kws={"label": "Per-class F1"},
        vmin=0.0,
        vmax=df[classes].values.max(),
    )
    ax.set_title("Per-class F1 per Experiment (sorted by macro avg)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Experiment")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    _save(fig, out_path)
    return fig


def print_classification_report(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    classes: list = CLASSES,
) -> None:
    """Collect val predictions and print sklearn classification report."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        zero_division=0,
    ))
    return all_labels, all_preds


# ── internal helpers ──────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_path: Path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)


# ── local smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile, os
    sys.path.insert(0, str(Path(__file__).parents[2]))

    import torch
    from src.config import CLASSES, OUT_DIR

    device = torch.device("cpu")
    out_dir = Path(tempfile.mkdtemp())

    # --- test plot_leaderboard with dummy tracker ---
    dummy_tracker = {
        "exp_A": {"val_macro_f1": 0.22, "train_time_s": 90.0},
        "exp_B": {"val_macro_f1": 0.24, "train_time_s": 120.0},
        "exp_C": {"val_macro_f1": 0.19, "train_time_s": 75.0},
    }
    fig = plot_leaderboard(dummy_tracker, out_dir / "leaderboard_test.png")
    plt.close(fig)
    assert (out_dir / "leaderboard_test.png").exists()
    print("[PASS] plot_leaderboard saved figure")

    # --- test compute_per_class_f1 with a trivial model ---
    from torch.utils.data import TensorDataset

    n_classes = len(CLASSES)
    dummy_imgs   = torch.randn(20, 3, 64, 64)
    dummy_labels = torch.randint(0, n_classes, (20,))
    ds = TensorDataset(dummy_imgs, dummy_labels)
    loader = DataLoader(ds, batch_size=10)

    # model that always predicts class 0
    class ZeroModel(nn.Module):
        def forward(self, x):
            return torch.zeros(x.size(0), n_classes)

    f1_dict = compute_per_class_f1(ZeroModel(), loader, device)
    assert set(f1_dict.keys()) == set(CLASSES)
    print(f"[PASS] compute_per_class_f1 returned {len(f1_dict)} classes")

    # --- test print_classification_report ---
    labels_out, preds_out = print_classification_report(ZeroModel(), loader, device)
    assert len(labels_out) == 20
    print("[PASS] print_classification_report ran without error")

    print("\nAll analysis.py smoke tests passed.")
