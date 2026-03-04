# PROJ_PLAN.md — Pokémon Image Classification (DL Assignment 1)

> **Goal:** Classify Pokémon images into 9 types using MLP → CNN → Transfer Learning.
> **Metric:** Macro-averaged F1 (used because of class imbalance).
> **Budget:** ≤ 5 h GPU total on Google Colab free tier (1h MLP, 1h CNN, 3h Transfer).
> **Data:** 3600 train / 900 test, PNG images, 9 classes, imbalanced.

---

## Agent Guidelines (read before implementing anything)

- Read this file and all existing `src/` files before writing any code. Avoid duplication.
- Think as a software architect first. Before implementing, think the 3 best options at project level and at code level, then choose the best.
- Simple and clean over clever. No over-engineering. No "might be useful later" functions.
- Use what PyTorch / sklearn / torchvision already give you. Don't reimplement what libraries handle.
- Short comment above each function (one line). Inline comments in natural language, casual, like one dev explaining to another. No emojis anywhere (terminal encoding issues).
- Logs: only what's needed to pinpoint errors. Single-line, concise.
- Each file has an `if __name__ == "__main__"` block with local tests. Test on CPU with small data first, then move to Colab + GPU.
- Terminal is PowerShell: use `;` not `&&` to chain commands.
- When done with a task, give a short chat summary of what was implemented, in which file, and the data flow. No summary files.

---

## Project Structure

```
assignment_1/
|
├── data/                          # NOT committed — download from Google Drive (see notebook Cell 2)
|   ├── Train/                     # 3600 PNG images (UUID filenames)
|   ├── Test/                      # 900 PNG images
|   └── train_labels.csv           # UUID,label pairs
|
├── src/
|   ├── config.py                  # all hyperparams and paths — single source of truth
|   |
|   ├── datasets/
|   |   ├── dataset.py             # PokemonDataset class + transforms + class weights
|   |   ├── eda.py                 # EDA stats: class counts, imbalance ratios, image info
|   |   ├── eda_plots.py           # EDA visualizations: distribution, sample grid, pixel stats
|   |   └── dataset_test.py        # local tests: shapes, labels, splits, transforms
|   |
|   ├── training/
|   |   ├── train.py               # shared train_one_epoch + evaluate (used by all 3 tasks)
|   |   ├── early_stopping.py      # EarlyStopping class
|   |   └── train_test.py          # local tests: loss decreases, output shapes
|   |
|   ├── models/
|   |   ├── mlp.py                 # MLP model definition
|   |   ├── cnn.py                 # custom CNN model definition  [Task 2]
|   |   ├── transfer.py            # EfficientNet-B0 wrapper      [Task 3]
|   |   └── models_test.py         # local tests: forward pass shapes for all 3 models
|   |
|   └── evaluation/
|       ├── metrics.py             # compute_macro_f1, per-class report
|       ├── plots.py               # plot_history, plot_confusion_matrix
|       └── submission.py          # generate_submission -> CSV + format validation
|
├── task1/
|   └── notebook.ipynb             # Task 1 notebook: EDA + train MLP + evaluate + submit
|                                  # The notebook IS the orchestrator — all logic is inline
|
├── outputs/                       # auto-created at runtime, NOT committed
|   ├── checkpoints/               # best model .pth files per task
|   ├── results/                   # submission CSVs
|   └── plots/                     # all saved figures
|
├── .gitignore
├── requirements.txt
└── README.md
```

**Key principle:** All reusable logic lives in `src/`. The notebook is both the runner and the readable story — students can follow each cell and understand the full training flow. No separate `task*.py` entry-point scripts.

**Note on `src/datasets/` naming:** `src/data/` caused a namespace collision on Colab — Python's implicit namespace packages registered `assignment_1/data/` (the raw dataset folder) as the top-level `data` module, shadowing `src.data`. Renamed to `src/datasets/` to eliminate the collision permanently.

---

## Config (`src/config.py`) — DONE

Single file with all hyperparameters and paths. Every other file imports from here — no magic numbers anywhere else.

```python
SEED        = 42
FAST_RUN    = True   # 4 epochs / patience 2 for pipeline testing; set False for real runs
BATCH_SIZE  = 64
EPOCHS      = 4 if FAST_RUN else 30
LR          = 1e-3
PATIENCE    = 2 if FAST_RUN else 5
NUM_WORKERS = 2

IMG_SIZE_SMALL  = 64    # MLP and CNN
IMG_SIZE_LARGE  = 224   # Transfer Learning

NUM_CLASSES = 9
CLASSES     = ["Bug", "Fighting", "Fire", "Grass", "Ground", "Normal", "Poison", "Rock", "Water"]

DATA_DIR    = Path("data")
OUT_DIR     = Path("outputs")
```

---

## Data Pipeline (`src/datasets/dataset.py`) — DONE

**`PokemonDataset(img_dir, transform, csv_path=None, indices=None)`**
- Training mode (`csv_path` given): returns `(tensor, int_label)`
- Inference mode (`csv_path=None`): returns `(tensor, uuid_string)` — used for the test set

**Splits** — stratified 80/20 train/val using `sklearn.model_selection.train_test_split(stratify=y)`.

**Transforms:**
- `get_base_transforms(size)` — Resize + ToTensor + Normalize (ImageNet mean/std)
- `get_augment_transforms(size)` — adds RandomHorizontalFlip, ColorJitter, RandomRotation(15)

**Class weights** — `compute_class_weights(labels)` returns `total / (num_classes * class_count)` — rarer classes get higher weight. Passed to `CrossEntropyLoss(weight=...)`.

**`use_sampler`** — Task 1 (MLP): `False` — 2.76× imbalance is mild enough that weighted loss alone handles it. Tasks 2 and 3: revisit after training.

---

## EDA (`src/datasets/eda.py` + `eda_plots.py`) — DONE

`eda.py` computes and prints stats (no plots). `eda_plots.py` saves figures to `outputs/plots/` and returns the `plt.Figure` for inline display.

**`eda.py`:** `class_distribution`, `image_size_distribution`, `check_data_integrity`

**`eda_plots.py`:** `plot_class_distribution`, `plot_sample_images`, `plot_average_image_per_class`, `plot_pixel_statistics`, `plot_pixel_intensity_histogram`

---

## Task 1 — MLP (`src/models/mlp.py`) — DONE

Architecture: `Flatten → Linear(12288, 512) → BN → ReLU → Dropout(0.4) → Linear(512, 256) → BN → ReLU → Dropout(0.4) → Linear(256, 128) → BN → ReLU → Dropout(0.4) → Linear(128, 9)`

- ~6.4M parameters. Input size: 64×64×3 = 12,288 (flat pixel vector).
- Loss: weighted CrossEntropy. Optimizer: Adam (lr=1e-3). Scheduler: StepLR (step_size=5, γ=0.5). EarlyStopping(patience=5).
- Orchestrated by `task1/notebook.ipynb` — full inline flow (EDA → loaders → train → evaluate → plots → submission).

---

## Task 2 — CNN (`src/models/cnn.py`) — TODO

Architecture: 3 conv blocks → Global Average Pooling → FC head

```
Conv2d(3,32)   → BN → ReLU → MaxPool
Conv2d(32,64)  → BN → ReLU → MaxPool
Conv2d(64,128) → BN → ReLU → MaxPool
GlobalAvgPool → Dropout(0.5) → FC(128, 9)
```

- GlobalAvgPool instead of Flatten: fewer params, less overfitting, spatially robust.
- Data augmentation via `get_augment_transforms`. Image size: 64×64.
- Same train loop as Task 1 (shared `src/training/train.py`).
- Orchestrated by `task2/notebook.ipynb` (to be created).

---

## Task 3 — Transfer Learning (`src/models/transfer.py`) — TODO

**Backbone:** EfficientNet-B0 from `torchvision.models` (pre-trained ImageNet). Best accuracy/param tradeoff, fits in 3 h budget.

**Fine-tuning strategy (two phases):**
1. Freeze backbone → train classifier head only (10 epochs, lr=1e-3).
2. Unfreeze last 2 blocks → train at lower lr (lr=1e-4, 15 epochs).

- Input size: 224×224. Full augmentation. AMP via `torch.amp.GradScaler` (halves memory, ~2× faster).
- Orchestrated by `task3/notebook.ipynb` (to be created).

---

## Shared Training Loop (`src/training/train.py`) — DONE

All 3 tasks use the same two functions:

```python
train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float
evaluate(model, loader, criterion, device) -> {"loss": float, "acc": float, "macro_f1": float}
```

`scaler=None` for Tasks 1 and 2 (FP32). `scaler=GradScaler()` for Task 3 (AMP). Per-epoch logging lives in the notebook cells — one print line per epoch.

---

## Evaluation (`src/evaluation/`) — DONE

**`metrics.py`:** `compute_macro_f1`, `classification_report_str`

**`plots.py`:** `plot_history(history, out_path)`, `plot_confusion_matrix(y_true, y_pred, classes, out_path)`
`history` dict keys: `train_loss`, `val_loss`, `val_f1`.

**`submission.py`:** `generate_submission(model, test_loader, label_map, out_path, device)`, `validate_submission(path, expected_rows=900)`

---

## Import System — How It Works

No `pip install -e .`, no `pyproject.toml`, no `setup.cfg`.

All scripts and notebooks work by having `assignment_1/` on `sys.path`. Run scripts from `assignment_1/` root:

```powershell
python -m src.datasets.dataset_test
python -m src.training.train_test
python -m src.models.models_test
```

The notebook setup cell adds the root to `sys.path` explicitly:
```python
ROOT = os.getcwd()   # /content/DL_Proj/assignment_1
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```

Intra-package imports (inside `src/`) use relative imports (`from ..config import ...`).
Entry-point imports (notebook) use absolute imports (`from src.datasets.dataset import ...`).

---

## Local Testing Strategy

Each `*_test.py` runs standalone on CPU.

**`dataset_test.py`** — dataset length (3600), tensor shape `(3, H, W)`, labels in `[0, 8]`, stratified split ratios, class weights shape, inference mode (returns uuid strings).

**`models_test.py`** — forward pass of MLP with `torch.randn(4, 3, 64, 64)` → shape `(4, 9)`. CNN and Transfer added as those files are implemented.

**`train_test.py`** — loss decreases after 10 steps on tiny batch; `evaluate()` returns correct keys and value ranges.

---

## Notebook Design

Each task has its own notebook (`task1/notebook.ipynb`, `task2/notebook.ipynb`, `task3/notebook.ipynb`). The notebook is both the runner and the readable story — every step is an inline cell so a student can read, run, and understand the full flow.

Cell structure per task:
1. **Setup** (run once per session): clone repo + install deps + `sys.path`
2. **Data** (run once): download from Google Drive if not present
3. **EDA cells** (Task 1 only): stats + 5 plots inline
4. **Data loaders + model setup**
5. **Training loop** (inline — explicit for/epoch with print)
6. **Load best checkpoint + classification report**
7. **Training curves + confusion matrix**
8. **Submission CSV generation + validation**

---

## Class Imbalance — Key Numbers

```
Water:    674  (18.7%)   <- majority
Normal:   606  (16.8%)
Poison:   467  (13.0%)
Fire:     381  (10.6%)
Bug:      374  (10.4%)
Grass:    299   (8.3%)
Fighting: 291   (8.1%)
Rock:     264   (7.3%)
Ground:   244   (6.8%)   <- minority (2.76x less than Water)
```

Imbalance ratio: 2.76×. Mitigation: weighted CrossEntropy + stratified split. WeightedRandomSampler optional from Task 2 onward.

---

## Requirements (`requirements.txt`)

```
torch>=2.2
torchvision>=0.17
scikit-learn>=1.4
pandas>=2.0
matplotlib>=3.8
seaborn>=0.13
Pillow>=10.0
tqdm>=4.66
```

---

## Implementation Order

### Foundation — ALL DONE ✓
- [x] `src/config.py`
- [x] `src/datasets/dataset.py` + `dataset_test.py`
- [x] `src/evaluation/metrics.py` + `submission.py`
- [x] `src/training/train.py` + `early_stopping.py` + `train_test.py`
- [x] `src/evaluation/plots.py`

### Task 1 — DONE ✓
- [x] `src/datasets/eda.py` + `eda_plots.py`
- [x] `src/models/mlp.py` + `models_test.py`
- [x] `task1/notebook.ipynb` — full inline flow

### Task 2 — TODO
- [ ] `src/models/cnn.py` (extend `models_test.py`)
- [ ] `task2/notebook.ipynb`

### Task 3 — TODO
- [ ] `src/models/transfer.py` (extend `models_test.py`)
- [ ] `task3/notebook.ipynb`

### Workflow
```
1. implement src/ file
2. run local tests (CPU): python -m src.<pkg>.<test_file>
3. fix until tests pass
4. add cells to task notebook, run on Colab GPU
5. save outputs and submission CSV
```
