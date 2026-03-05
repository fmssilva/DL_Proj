# Shared constants and helpers used by all tasks and src/ modules.
# Hyperparameters (epochs, lr, batch size, etc.) live in each task's notebook —
# they are run-level decisions, not project-level ones.

import random
from pathlib import Path

import numpy as np
import torch

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── classes — single source of truth for label ordering ──────────────────────
# index i maps to CLASSES[i] everywhere; no separate label encoder needed
NUM_CLASSES = 9
CLASSES = ["Bug", "Fighting", "Fire", "Grass", "Ground", "Normal", "Poison", "Rock", "Water"]

# ── canonical data paths (relative to assignment_1/ root) ────────────────────
DATA_DIR  = Path("data")
OUT_DIR   = Path("outputs")   # kept for any global/shared fallback use


# ── helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn deterministic — slight perf cost, worth it for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def get_task_out_dir(task_name: str) -> Path:
    """
    Return base output dir for a given task (e.g. 'task1') and create subfolders.
    Each task writes its own checkpoints/plots/results so outputs never mix across tasks.
    Layout: <task_name>/outputs/{checkpoints,plots,results}/
    """
    base = Path(task_name) / "outputs"
    for subdir in ["checkpoints", "plots", "results"]:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base
