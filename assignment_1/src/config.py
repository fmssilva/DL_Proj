# Single source of truth for all hyperparameters and paths.
# Every other file imports from here — no magic numbers anywhere else.

import random
from pathlib import Path

import numpy as np
import torch

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── fast-run flag ─────────────────────────────────────────────────────────────
# True  -> 2 epochs, patience 1 — just enough to verify the pipeline runs end-to-end.
# False -> 30 epochs, patience 5 — use this on Colab GPU for the real training run.
FAST_RUN = True

# ── training ─────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 2  if FAST_RUN else 30
LR          = 1e-3
PATIENCE    = 1  if FAST_RUN else 5
NUM_WORKERS = 2

# ── image sizes ───────────────────────────────────────────────────────────────
IMG_SIZE_SMALL = 64   # MLP + CNN
IMG_SIZE_LARGE = 224  # Transfer Learning (EfficientNet expects 224)

# ── classes ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 9
# sorted alphabetically — index i maps to CLASSES[i] everywhere (no separate encoder)
CLASSES = ["Bug", "Fighting", "Fire", "Grass", "Ground", "Normal", "Poison", "Rock", "Water"]

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes cudnn deterministic — slight perf cost, worth it for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_output_dirs() -> None:
    """Create the outputs/ folder tree if it doesn't already exist."""
    for subdir in ["checkpoints", "results", "plots"]:
        (OUT_DIR / subdir).mkdir(parents=True, exist_ok=True)
