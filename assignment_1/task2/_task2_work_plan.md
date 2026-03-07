# Task 2 — CNN Work Plan

> For an AI agent implementing this from scratch. Read fully before touching any code.
> Last updated: clean architecture sweep (no aug A–F), then aug experiments, fixed build_loaders (train_transform param), +LR scheduler experiment, fixed ensemble expected-results framing, added debugging checklist.

---

## 0. Context — What Already Exists (Don't Duplicate)

All of this is shared infrastructure reused verbatim in Task 2:

| What                                                                             | Where                            | How task2 uses it                                    |
| -------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------- |
| `PokemonDataset`, all transforms, `get_train_val_loaders`                        | `src/datasets/dataset.py`        | Same `build_loaders()` helper in notebook            |
| `train_one_epoch`, `evaluate`                                                    | `src/training/train.py`          | Same `run_experiment()` inline function              |
| `EarlyStopping`                                                                  | `src/training/early_stopping.py` | Same `patience` logic                                |
| `save_experiment_result`, `restore_tracker`, `ResultsTracker`, `ExperimentEntry` | `src/evaluation/persistence.py`  | Same `results_tracker: ResultsTracker = {}`          |
| `plot_leaderboard`, `plot_per_class_f1_heatmap`, `print_classification_report`   | `src/evaluation/analysis.py`     | Called identically                                   |
| `soft_ensemble`, `print_ensemble_report`                                         | `src/evaluation/ensemble.py`     | Works with any architecture mix — CNN+CNN or CNN+MLP |
| `generate_submission_from_preds`, `validate_submission`                          | `src/evaluation/submission.py`   | Same auto-submission logic in `run_experiment`       |
| `CLASSES`, `NUM_CLASSES`, `SEED`, `set_seed`, `get_task_out_dir`                 | `src/config.py`                  | Same imports                                         |

**The only new file to create:** `src/models/cnn.py`  
**The only test file to extend:** `src/models/models_test.py` (add CNN forward pass tests)  
**The notebook:** copy `task1/task1_colab.ipynb` → `task2/task2_colab.ipynb`, strip to structure, replace MLP imports/experiments with CNN

---

## 1. New File: `src/models/cnn.py`

### Design principles
- No softmax — `CrossEntropyLoss` handles it
- All classes: `dropout=0.5` default (higher than MLP's 0.3-0.4 because GlobalAvgPool compresses to a small vector — but notebook can override per experiment)
- `GlobalAvgPool2d` replaces `Flatten` — far fewer params, less overfit, spatially robust
- Each conv block: `Conv2d(k=3, p=1) → BN → ReLU → MaxPool2d(2,2)`
- No Dropout inside conv blocks — only in the FC head after GlobalAvgPool
- Input: `(B, 3, 64, 64)` RGB tensors. Output: `(B, 9)` logits.

### Classes to implement (in order, from baseline to best)

#### `BaseCNN` — 3-block baseline (~120K params)
```
Conv(3→32,  k=3, p=1) → BN → ReLU → MaxPool(2)  # 64→32
Conv(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)  # 32→16
Conv(64→128,k=3, p=1) → BN → ReLU → MaxPool(2)  # 16→8
GlobalAvgPool → [128]
Dropout(p) → Linear(128, 9)
```
Clean reference point — every other experiment compares back to this.

#### `BaseCNN_NoBN` — BaseCNN without BatchNorm (ablation only)
Identical to `BaseCNN` but with all `BatchNorm2d` layers removed. Used in experiment L to quantify exactly how many F1 points BN contributes. Not intended as a production model.

#### `DeepCNN` — 4 blocks, more depth
```
Conv(3→32)   → BN → ReLU → MaxPool  # 64→32
Conv(32→64)  → BN → ReLU → MaxPool  # 32→16
Conv(64→128) → BN → ReLU → MaxPool  # 16→8
Conv(128→256)→ BN → ReLU → MaxPool  # 8→4
GlobalAvgPool → [256]
Dropout(p) → Linear(256, 9)
```
Tests depth before spatial map collapses (4×4 after 4 pools on 64px input is the limit).

#### `WideCNN` — 3 blocks, wider channels
```
Conv(3→64)   → BN → ReLU → MaxPool  # 64→32
Conv(64→128) → BN → ReLU → MaxPool  # 32→16
Conv(128→256)→ BN → ReLU → MaxPool  # 16→8
GlobalAvgPool → [256]
Dropout(p) → Linear(256, 9)
```
More filters per layer = richer feature maps. Same depth as BaseCNN, directly comparable.

#### `SEBlock` — channel attention helper (used inside ResidualCNN)
```python
class SEBlock(nn.Module):
    # squeeze: GlobalAvgPool → [C]
    # excite: FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid
    # scale: multiply original feature map channel-wise
    def __init__(self, channels, reduction=8): ...
```
`reduction=8` keeps it lightweight (~2× channels params overhead). Plugged into `ResidualCNN` optionally via `use_se=True`.

#### `ResidualCNN` — 3 residual blocks + optional SE attention
Each residual block:
```
input → Conv(k=3,p=1) → BN → ReLU → Conv(k=3,p=1) → BN
      + 1×1 projection (when in_ch ≠ out_ch)   ← skip
→ ReLU → [optional SEBlock] → MaxPool(2)
```
Full architecture:
```
ResBlock(3→64,   se=False/True) → MaxPool  # 64→32
ResBlock(64→128, se=False/True) → MaxPool  # 32→16
ResBlock(128→256,se=False/True) → MaxPool  # 16→8
GlobalAvgPool → Dropout(p) → Linear(256, 9)
```
`use_se=False` = plain `ResidualCNN`. `use_se=True` = `ResidualCNN` with SE attention.
Both instantiated from the same class — no code duplication.

**Projection conv clarification:** The 1×1 projection fires **only when `in_channels != out_channels`** (i.e. the first ResBlock 3→64 and subsequent blocks when channels grow). When `in_channels == out_channels`, the identity shortcut is used directly — zero extra parameters. A common bug is applying projection unconditionally; gate it with `if in_ch != out_ch`.

#### `MultiScaleCNN` — parallel k=3 and k=5 branches per block
```
Block 1: [Conv(3→16,k=3,p=1) || Conv(3→16,k=5,p=2)] → cat→32ch → BN → ReLU → MaxPool
Block 2: [Conv(32→32,k=3,p=1)|| Conv(32→32,k=5,p=2)]→ cat→64ch → BN → ReLU → MaxPool
Block 3: [Conv(64→64,k=3,p=1)|| Conv(64→64,k=5,p=2)]→ cat→128ch → BN → ReLU → MaxPool
GlobalAvgPool → Dropout(p) → Linear(128, 9)
```
Note: `k=5, p=2` preserves spatial size before MaxPool. Memory cost is ~2× BaseCNN. Run without augmentation first to isolate the architecture effect cleanly.

### Architecture docstring pattern (same as mlp.py)
Each class: short docstring + ASCII schema showing spatial dims at 64×64 input + param count.

---

## 2. Extend `src/models/models_test.py`

Add CNN tests at the bottom, after the existing MLP tests. Pattern is identical to what's already there:
```python
from src.models.cnn import BaseCNN, BaseCNN_NoBN, DeepCNN, WideCNN, ResidualCNN, MultiScaleCNN

def test_cnn_forward():
    x = torch.randn(4, 3, 64, 64)
    _check_forward(BaseCNN(),                  x, "BaseCNN")
    _check_forward(BaseCNN_NoBN(),             x, "BaseCNN_NoBN")
    _check_forward(DeepCNN(),                  x, "DeepCNN")
    _check_forward(WideCNN(),                  x, "WideCNN")
    _check_forward(ResidualCNN(use_se=False),  x, "ResidualCNN")
    _check_forward(ResidualCNN(use_se=True),   x, "ResidualCNN+SE")
    _check_forward(MultiScaleCNN(),            x, "MultiScaleCNN")
```
Add `test_cnn_forward()` to the **existing** `if __name__ == "__main__"` block — do not add a second one.  
Run with: `python -m src.models.models_test`
All `[PASS]`, output shape `(4, 9)`, no NaN. **Do not touch the notebook until this passes.**

---

## 3. `src/datasets/dataset.py` — Two changes

### 3a. Add `get_strong_aug_transforms`

Add this function after `get_augment_transforms`. **Do not modify any existing function.**

```python
def get_strong_aug_transforms(size: int) -> transforms.Compose:
    """Aggressive RGB augmentation for CNN experiment H. Includes RandomErasing and stronger jitter."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),           # Pokémon sprites appear in various orientations
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # small positional shifts
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # mask random rectangles — MUST be after ToTensor
    ])
```

`RandomErasing` **must** come after `ToTensor` — it operates on tensors, not PIL images. Wrong position = silent error, not a crash.

### 3b. Add `train_transform` parameter to `get_train_val_loaders`

This is a one-line change that makes `build_loaders` clean instead of relying on post-hoc mutation:

```python
def get_train_val_loaders(
    csv_path, img_dir, img_size, batch_size,
    augment=False, use_sampler=False, num_workers=2,
    df_override=None, grayscale=False, equalize=False,
    train_transform=None,   # ← ADD THIS (overrides augment flag if provided)
) -> tuple[DataLoader, DataLoader]:
    ...
    # in the transform selection block, add at the end:
    if train_transform is not None:
        pass  # use caller-supplied transform directly
    elif grayscale:
        train_transform = get_gray_aug_transforms(...) if augment else get_gray_transforms(...)
    else:
        train_transform = get_augment_transforms(img_size) if augment else get_base_transforms(img_size)
```

The val transform is **never** overridden — always `get_base_transforms` (or grayscale equivalent).

### 3c. Test `get_strong_aug_transforms` locally

Add to `src/datasets/dataset_test.py` (or inline before Colab):

```python
from PIL import Image
from src.datasets.dataset import get_strong_aug_transforms

t = get_strong_aug_transforms(64)
out = t(Image.new("RGB", (100, 100)))
assert out.shape == (3, 64, 64), f"shape wrong: {out.shape}"
assert out.dtype == torch.float32, f"dtype wrong: {out.dtype}"
print("[PASS] get_strong_aug_transforms")
```

This catches the common mistake of placing `RandomErasing` before `ToTensor` (which raises a TypeError at runtime).

---

## 4. Notebook: `task2/task2_colab.ipynb`

### How to create it
Copy `task1/task1_colab.ipynb` → `task2/task2_colab.ipynb`. Then:
- Keep all infrastructure cells (Colab setup, Drive mount, data download, EDA — all identical)
- Keep the Results Manager cell verbatim — `ResultsTracker`, `_print_leaderboard`, `restore_tracker` unchanged
- **Delete** all individual experiment cells (A–S, grayscale section, augmentation section, ensemble runs)
- **Replace** the experiment setup cell with CNN imports + `build_loaders` + `model_registry`
- Add skeleton experiment cells A, B, C as placeholder structure
- Update `TASK_NAME = "task2"`, `TASK_OUT_DIR = get_task_out_dir("task2")`

### Key notebook changes vs task1

**Setup cell:**
```python
TASK_NAME  = "task2"
IMG_SIZE   = 64
EPOCHS     = 40    # CNNs converge slower than MLPs
PATIENCE   = 7     # CNN val curves are less monotone, need more patience
LR         = 1e-3
BATCH_SIZE = 64
```
> EarlyStopping with `patience=7` can extend runs past `EPOCHS=40` — it fires at `best_epoch + 7`, which may exceed EPOCHS. This is expected behaviour, not a bug. Maximum observed run length: ~47 epochs ≈ 165s on T4. Still well within the 1h Colab budget across all 13 solo experiments.

**Imports cell — replace MLP with CNN, keep MLP for ensemble:**
```python
from src.models.cnn import BaseCNN, BaseCNN_NoBN, DeepCNN, WideCNN, ResidualCNN, MultiScaleCNN
from src.models.mlp import MLP   # kept for cross-task ensemble cell only
from src.datasets.dataset import get_strong_aug_transforms  # new in task2
```

**`build_loaders` helper — clean `train_transform` pass-through:**
```python
def build_loaders(augment=False, use_sampler=False, strong_aug=False):
    """Picks the right train transform and passes it explicitly to get_train_val_loaders."""
    train_t = (
        get_strong_aug_transforms(IMG_SIZE) if strong_aug
        else get_augment_transforms(IMG_SIZE) if augment
        else None  # get_train_val_loaders uses get_base_transforms by default
    )
    return get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE, BATCH_SIZE,
        augment=False,          # always False — train_t handles augmentation explicitly
        use_sampler=use_sampler,
        num_workers=NUM_WORKERS, df_override=df,
        train_transform=train_t,
    )
```
`augment=False` is passed unconditionally — `train_t` already encodes the augmentation choice. Passing `augment=augment` when `strong_aug=True` would be harmless (train_transform overrides it) but confusing.

**`model_registry` cell — CNN factories:**
```python
model_registry = {
    "A_base_cnn":   {"model": lambda: BaseCNN(dropout=0.5),              "criterion": lambda: nn.CrossEntropyLoss(weight=class_weights)},
    "B_base_aug":   {"model": lambda: BaseCNN(dropout=0.5),              "criterion": lambda: nn.CrossEntropyLoss(weight=class_weights)},
    ...
    "G_residual":   {"model": lambda: ResidualCNN(dropout=0.5, use_se=False), "criterion": lambda: nn.CrossEntropyLoss(weight=class_weights)},
    "I_residual_se":{"model": lambda: ResidualCNN(dropout=0.5, use_se=True),  "criterion": lambda: nn.CrossEntropyLoss(weight=class_weights)},
    ...
}
```

**`run_experiment` function:** copy verbatim from task1 — zero changes needed.

**No grayscale section** — skip entirely.

---

## 5. Experiment Plan — What to Run in Colab

Run in strict order. `patience=7`. Estimated ~150s/run on T4 → ~16 runs × 150s ≈ 40 min training.

**Design principle:** Architecture sweep first on clean data (no aug A–F), so every comparison is one variable at a time. Only after the best architecture is identified do we add augmentation (G, H) and ablations (K–N).

### Phase 1 — Architecture sweep (no augmentation)

| ID  | Name            | Architecture          | Aug | Sampler | Loss       | What it answers                            |
| --- | --------------- | --------------------- | --- | ------- | ---------- | ------------------------------------------ |
| A   | `A_base_cnn`    | BaseCNN, drop=0.5     | No  | No      | CE+weights | Baseline: beat MLP (0.243 target)?         |
| B   | `B_wide`        | WideCNN, drop=0.5     | No  | No      | CE+weights | Width vs depth: more filters per layer     |
| C   | `C_deep`        | DeepCNN, drop=0.5     | No  | No      | CE+weights | Depth: 4 blocks (spatial collapses to 4×4) |
| D   | `D_residual`    | ResidualCNN, se=False | No  | No      | CE+weights | Skip connections — expected biggest jump   |
| E   | `E_residual_se` | ResidualCNN, se=True  | No  | No      | CE+weights | SE attention on top of best arch so far    |
| F   | `F_multiscale`  | MultiScaleCNN         | No  | No      | CE+weights | Multi-scale texture receptive fields       |

→ **After F: identify best architecture from the leaderboard.** Call it `BEST_ARCH` in subsequent cells.

### Phase 2 — Augmentation on best architecture

| ID  | Name                | Architecture | Aug            | Sampler | Loss       | What it answers                           |
| --- | ------------------- | ------------ | -------------- | ------- | ---------- | ----------------------------------------- |
| G   | `G_best_aug`        | BEST_ARCH    | Yes (standard) | No      | CE+weights | Does standard aug help CNN? By how much?  |
| H   | `H_best_strong_aug` | BEST_ARCH    | Strong         | No      | CE+weights | How much further does strong aug push it? |

> **G and H are defined after running A–F.** Before running G, look at the leaderboard and substitute the winning architecture. If `D_residual` wins, `G_best_aug = ResidualCNN(use_se=False)`. Do not pre-fill this cell.

### Phase 3 — Regularisation & loss variants (best arch + standard aug)

| ID  | Name             | Architecture | Aug | Sampler | Loss               | What it answers                           |
| --- | ---------------- | ------------ | --- | ------- | ------------------ | ----------------------------------------- |
| I   | `I_best_sampler` | BEST_ARCH    | Yes | Yes     | CE (no weights)    | Sampler vs weighted loss for imbalance    |
| J   | `J_best_ls`      | BEST_ARCH    | Yes | No      | CE+weights, LS=0.1 | Label smoothing (confirmed +0.03 for MLP) |

### Phase 4 — Ablations (no augmentation for clean comparisons)

| ID  | Name           | Architecture          | Aug | Sampler | Loss       | What it answers                                      |
| --- | -------------- | --------------------- | --- | ------- | ---------- | ---------------------------------------------------- |
| K   | `K_drop_sweep` | ResidualCNN, drop=0.3 | No  | No      | CE+weights | Dropout 0.3 vs 0.5 on 256-dim head (production arch) |
| L   | `L_base_no_bn` | BaseCNN_NoBN          | No  | No      | CE+weights | Quantify BN contribution in F1 points                |
| M   | `M_cosine_lr`  | ResidualCNN, se=False | No  | No      | CE+weights | CosineAnnealingLR vs StepLR on CNN                   |

> **K uses ResidualCNN** (not BaseCNN) — dropout sensitivity on a 256-dim head may differ from 128-dim; test it on the architecture you'll actually use.  
> **L uses BaseCNN** — isolating BN is cleanest on the simplest architecture; the delta is architecture-agnostic.  
> **M (LR scheduler):** Run as pure information only — by Phase 4, G–J have already run with StepLR. Do not retroactively re-run them. Report: "CosineAnnealing yields +X F1 on clean ResidualCNN; we kept StepLR for consistency across all aug experiments." Task1 found no improvement for MLP; CNN may differ due to deeper gradient flow.

### Ensembles

| ID          | Name     | Components                | What it answers                      |
| ----------- | -------- | ------------------------- | ------------------------------------ |
| ENS_best2   | Ensemble | top-2 CNNs by val F1      | Does averaging top-2 beat solo best? |
| ENS_cnn_mlp | Ensemble | best CNN + task1 best MLP | Cross-task diversity (expected: no)  |

### Notes
- **I uses CE without class weights** — sampler already rebalances the batch distribution; combining both over-compensates (Task 1 experiment O confirmed this: F1=0.192, one of the worst results). When using sampler, drop weights.
- **K and L (ablations):** No augmentation — same clean conditions as Phase 1 for a fair comparison.
- **ENS_cnn_mlp:** Almost certainly MLP drags the ensemble down. Run it anyway — "we confirmed it doesn't help" is a credible finding.
- **Phase 2 BEST_ARCH:** If results are very close (within 0.01 F1), pick simpler arch (prefer ResidualCNN over MultiScaleCNN for interpretability).

---

## 6. Evaluation & Comparison Cells (after all experiments)

Same structure as task1 Part 4. In order:

1. **Leaderboard table** — `_print_leaderboard(results_tracker)` — no changes needed
2. **Per-class F1 heatmap** — `plot_per_class_f1_heatmap(checkpoint_dir, model_registry, build_loaders, device, out_path)`
3. **Best model deep dive** — `print_classification_report` + `plot_history` + `plot_confusion_matrix`
4. **MLP vs CNN comparison** — load task1 JSON, side-by-side table:
```python
from src.evaluation.persistence import load_results
task1_data    = load_results(Path("task1/outputs/results/task1_results.json"))
task1_best_f1 = task1_data.get("best_val_macro_f1", "N/A")
task2_best_f1 = max(
    v["val_macro_f1"] for k, v in results_tracker.items() if not k.startswith("ENS_")
)
print(f"Task1 best MLP : {task1_best_f1:.4f}")
print(f"Task2 best CNN : {task2_best_f1:.4f}")
print(f"Delta          : {task2_best_f1 - float(task1_best_f1):+.4f}")
```
5. **Ensemble cells** — CNN+CNN, then CNN+MLP cross-task (use `soft_ensemble` + `print_ensemble_report` as in task1)
6. **Submission** — auto-generated inside `run_experiment`; add a final `validate_submission(SUB_PATH)` cell

---

## 7. Expected Results & Decision Points

- **A_base_cnn** should beat MLP (~0.243 → ~0.35+). If it doesn't, debug with this checklist:
  - [ ] Val loader uses `get_base_transforms` (not augmented) — check the DataLoader passed to `evaluate()`
  - [ ] Class weights computed from **train split only**, not full dataset
  - [ ] Model in `model.train()` during training, `model.eval()` during evaluation
  - [ ] EarlyStopping monitoring `-val_macro_f1`, not `val_loss`
- **Architecture sweep (A–F):** Each experiment is one variable vs A. Expect `D_residual` to show the biggest jump. If `E_residual_se` beats `D_residual` cleanly (>+0.02), SE is worth the complexity; if delta is <0.01, the simpler model wins.
- **G vs A augmentation delta**: expect +0.03 to +0.05 F1. CNN preserves spatial layout — flips and jitter give genuine new training signal.
- **H strong aug vs G**: expect smaller delta (+0.01–0.03). Strong aug is regularisation at this dataset size; beyond a point it adds noise faster than signal.
- **K_drop_sweep**: if GlobalAvgPool compresses to 128-dim, dropout=0.3 may outperform 0.5 — or not. Either result is a clean data point.
- **L_base_no_bn**: expect BN to contribute +0.04–0.08 F1. Without BN, training is slower and less stable on small datasets.
- **M_cosine_lr**: expect small delta (±0.02). Task1 found no improvement for MLP; CNN may differ because of deeper gradient flow. Worth one run.
- **ENS_cnn_mlp**: do not expect this to beat solo CNN. MLP and CNN receive the same image pixels but produce very different probability distributions — MLP's lower confidence on most samples means it adds noise, not diversity. The soft ensemble averages probability vectors, not F1 scores; the CNN's sharper distributions will dominate, but the MLP's weaker signal on its correct predictions dilutes the CNN's correct predictions. "We confirmed it doesn't help" is a valid finding.
- **Target before Task 3**: val_macro_f1 ≥ 0.50. ResidualCNN + SE + augmentation should reach this.

---

## 8. Implementation Order (strict)

1. `src/models/cnn.py` — all 6 classes (BaseCNN, BaseCNN_NoBN, DeepCNN, WideCNN, ResidualCNN with SEBlock, MultiScaleCNN)
2. `src/models/models_test.py` — extend with CNN forward pass tests
3. Run locally: `python -m src.models.models_test` — all `[PASS]` before continuing
4. `src/datasets/dataset.py` — (a) add `get_strong_aug_transforms`, (b) add `train_transform=None` param to `get_train_val_loaders`, (c) run the one-liner transform test from Section 3c
5. `task2/task2_colab.ipynb` — copy from task1, strip experiments, update task name/hyperparams, add CNN structure
6. Push to GitHub
7. Open Colab, run Phase 1 (A–F, no aug) → look at leaderboard → run Phase 2 (G–H, best arch) → Phase 3 (I–J) → Phase 4 ablations (K–M) → ensembles → analysis + submission cells

---

## 9. What NOT to do

- Do not change `train.py`, `early_stopping.py`, `persistence.py`, `submission.py`, `ensemble.py` — task-agnostic, already tested
- Do not use `Flatten` + large FC in CNN — defeats the purpose; `GlobalAvgPool2d` is the right head
- Do not add Dropout inside conv blocks — only in the FC head after GlobalAvgPool
- Do not run augmentation experiments before the architecture sweep — augmentation confounds architecture comparisons; run A–F clean first
- Do not use `IMG_SIZE=128` — 1h Colab budget doesn't allow it; 64×64 is the right call
- Do not add a grayscale section — colour information is valuable for CNN; MLP already confirmed that story
- Do not mix `save_all_results` and `save_experiment_result` — use `save_experiment_result` per run in `run_experiment`, `save_all_results` only at the very end
- Do not mutate `train_loader.dataset.transform` post-hoc — use the `train_transform` parameter in `get_train_val_loaders` instead (added in Section 3b)
- Do not apply `RandomErasing` before `ToTensor` — it operates on tensors; wrong position silently corrupts values instead of raising an error
