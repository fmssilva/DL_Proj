# PROJ_PLAN_CLEAN.md — Pokémon Image Classification (DL Assignment 1)

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
├── data/                          # NOT committed — download from Kaggle
|   ├── Train/                     # 3600 PNG images (UUID filenames)
|   ├── Test/                      # 900 PNG images
|   └── train_labels.csv           # UUID,label pairs
|
├── src/
|   ├── config.py                  # all hyperparams and paths — single source of truth
|   |
|   ├── data/
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
|   |   ├── cnn.py                 # custom CNN model definition
|   |   ├── transfer.py            # EfficientNet-B0 wrapper + freeze/unfreeze helpers
|   |   └── models_test.py         # local tests: forward pass shapes for all 3 models
|   |
|   └── evaluation/
|       ├── metrics.py             # compute_macro_f1, per-class report
|       ├── plots.py               # plot_history, plot_confusion_matrix
|       └── submission.py          # generate_submission -> Kaggle CSV + format validation
|
├── task1_mlp.py                   # Task 1 entry-point: EDA + train MLP + evaluate + submit
├── task2_cnn.py                   # Task 2 entry-point: train CNN + compare with MLP
├── task3_transfer.py              # Task 3 entry-point: fine-tune EfficientNet + full report
|
├── notebook.ipynb                 # thin Colab runner: clone repo -> run tasks -> show plots
|
├── outputs/                       # auto-created at runtime, NOT committed
|   ├── checkpoints/               # best model .pth files per task
|   ├── results/                   # submission CSVs
|   └── plots/                     # all saved figures
|
├── assignment/                    # assignment brief and guides (already present)
├── .gitignore
├── requirements.txt
└── PROJ_PLAN.md
```

**Key principle:** All logic lives in `src/`. Entry-points (`task*.py`) are short orchestrators (~60 lines max). The notebook is only a launcher — it clones the repo and calls the task scripts. No logic in the notebook.

---

## Config (`src/config.py`)

Single file with all hyperparameters and paths. Every other file imports from here — no magic numbers anywhere else.

```python
SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 5
NUM_WORKERS = 2

IMG_SIZE_SMALL  = 64    # MLP and CNN
IMG_SIZE_LARGE  = 224   # Transfer Learning

NUM_CLASSES = 9
CLASSES     = ["Bug", "Fighting", "Fire", "Grass", "Ground", "Normal", "Poison", "Rock", "Water"]

DATA_DIR    = Path("data")
OUT_DIR     = Path("outputs")
```

---

## Data Pipeline (`src/datasets/dataset.py`)

**`PokemonDataset`** — torch Dataset that reads `train_labels.csv`, maps UUID to image path, and encodes labels as integers.

**Splits** — stratified 80/20 train/val using `sklearn.model_selection.train_test_split(stratify=y)`. Stratified so minority classes (Ground, Rock) appear in both splits proportionally.

**Transforms** — two presets, injected at construction time so the same Dataset class works for both:
- `get_base_transforms(size)` — Resize + ToTensor + Normalize (ImageNet mean/std, good default even for non-ImageNet data)
- `get_augment_transforms(size)` — adds RandomHorizontalFlip, ColorJitter, RandomRotation (used for CNN and Transfer)

**Class weights** — computed from label counts and passed to `CrossEntropyLoss(weight=...)` to handle imbalance. Formula: `total / (num_classes * class_count)` — rarer classes get higher weight.

**WeightedRandomSampler** — used alongside weighted loss; oversamples minority classes at DataLoader level. Useful for heavily imbalanced classes like Ground (244 samples vs Water's 674).

---

## EDA (`src/datasets/eda.py` and `src/datasets/eda_plots.py`)

`eda.py` computes and prints stats (no plots). `eda_plots.py` saves figures to `outputs/plots/`. Both are called from `task1_mlp.py`.

**`eda.py` functions:**
- `class_distribution(df)` — counts and percentages per class, imbalance ratio
- `image_size_distribution(img_dir)` — checks all image sizes, returns unique sizes found. Justifies resize choice.

**`eda_plots.py` functions:**
- `plot_class_distribution(df)` — bar chart with counts + percentages, highlights imbalance
- `plot_sample_images(dataset)` — grid of 3-5 sample images per class
- `plot_pixel_stats(dataset)` — per-channel mean/std across the dataset (justifies normalization values)

---

## Task 1 — MLP (`src/models/mlp.py` + `task1_mlp.py`)

Architecture: flatten image → FC stack `[Input → 512 → 256 → 128 → 9]`

- BatchNorm1d before each ReLU (stabilizes training on flat pixel inputs)
- Dropout(0.4) after each activation (heavy regularization needed — MLP will overfit fast)
- Loss: weighted CrossEntropy
- Optimizer: Adam (lr=1e-3)
- Scheduler: StepLR (step every 5 epochs, gamma=0.5)
- Image size: 64×64 (input dim = 64×64×3 = 12 288)
- EarlyStopping(patience=5) + save best checkpoint

`task1_mlp.py` flow: EDA → build loaders → train → evaluate → save submission

---

## Task 2 — CNN (`src/models/cnn.py` + `task2_cnn.py`)

Architecture: 3 conv blocks → Global Average Pooling → FC head

```
[Conv2d(3,32)   -> BN -> ReLU -> MaxPool]   # block 1
[Conv2d(32,64)  -> BN -> ReLU -> MaxPool]   # block 2
[Conv2d(64,128) -> BN -> ReLU -> MaxPool]   # block 3
GlobalAvgPool -> Dropout(0.5) -> FC(128, 9)
```

- GlobalAvgPool instead of Flatten: fewer params, less overfitting, spatially robust
- Data augmentation via `get_augment_transforms`
- Same train loop as Task 1 (shared `src/training/train.py`)
- Image size: 64×64

`task2_cnn.py` flow: build loaders → train → evaluate → compare metrics with Task 1 → save submission

---

## Task 3 — Transfer Learning (`src/models/transfer.py` + `task3_transfer.py`)

**Backbone:** EfficientNet-B0 from `torchvision.models` (pre-trained on ImageNet). Best accuracy/param tradeoff, compact enough to fine-tune within 3 h budget.

**Fine-tuning strategy (two phases):**
1. Freeze all backbone layers → train only the new classifier head (10 epochs, lr=1e-3). Lets the head adapt before touching backbone weights.
2. Unfreeze last 2 blocks → train everything at lower lr (lr=1e-4, 15 epochs). Gentle fine-tuning — high lr here would destroy pre-trained features.

**Other settings:**
- Input size: 224×224 (EfficientNet expects this)
- Full augmentation (`get_augment_transforms`)
- Mixed-precision training via `torch.cuda.amp` (halves memory, ~2x faster — critical for staying in 3 h budget)
- EarlyStopping(patience=5)

`task3_transfer.py` flow: build loaders → phase 1 train → phase 2 fine-tune → evaluate → compare all 3 tasks → save submission

---

## Shared Training Loop (`src/training/train.py`)

All 3 tasks use the same two functions — no code duplication:

```python
train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> avg_loss
evaluate(model, loader, criterion, device) -> {"loss": float, "acc": float, "macro_f1": float}
```

`scaler` is the AMP GradScaler for Task 3 mixed-precision. If `None`, runs normal FP32 (Tasks 1 and 2).

---

## Evaluation (`src/evaluation/`)

**`metrics.py`:**
- `compute_macro_f1(y_true, y_pred)` — wraps `sklearn.metrics.f1_score(average="macro")`
- `classification_report_str(y_true, y_pred)` — per-class precision/recall/F1 (shows which types are hard)

**`plots.py`:**
- `plot_history(history)` — loss and F1 curves over epochs, saved to `outputs/plots/`
- `plot_confusion_matrix(y_true, y_pred, classes)` — seaborn heatmap, saved to `outputs/plots/`

**`submission.py`:**
- `generate_submission(model, test_loader, label_encoder, path)` — runs inference, writes Kaggle CSV
- Validates before saving: correct row count (900), valid class names, correct header format

---

## Local Testing Strategy

Each `*_test.py` runs standalone on CPU with small synthetic or real data.

**`dataset_test.py`** checks:
- Dataset length matches CSV row count
- Image tensor shape is `(3, H, W)` with values in expected range after normalization
- Labels are integers in `[0, NUM_CLASSES-1]`
- Stratified split preserves class ratios (check with `Counter`)
- Class weights shape and no zeros

**`models_test.py`** checks:
- Forward pass of MLP with dummy input `torch.randn(4, 3, 64, 64)` returns shape `(4, 9)`
- Forward pass of CNN with same input returns shape `(4, 9)`
- Forward pass of Transfer model with `torch.randn(4, 3, 224, 224)` returns shape `(4, 9)`
- No NaN in any output

**`train_test.py`** checks:
- Loss decreases after 2-3 steps on a tiny batch (if not, something is broken)
- `evaluate` returns a dict with expected keys and value ranges `[0, 1]`

---

## Notebook (`notebook.ipynb`)

Thin Colab runner. No logic — just orchestration and inline display.

```python
# Cell 1: Setup (run once per session)
!git clone https://github.com/fmssilva/DL_Proj.git
%cd DL_Proj
!pip install -r requirements.txt
# Download data from Kaggle and place under data/
# from google.colab import files; files.upload()  <- upload kaggle.json
# !kaggle competitions download -c <competition-name>
# !unzip <competition-name>.zip -d data/

# Cell 2: Task 1 — MLP
!python task1_mlp.py
# display saved EDA and training plots inline

# Cell 3: Task 2 — CNN
!python task2_cnn.py
# display comparative training curves

# Cell 4: Task 3 — Transfer Learning
!python task3_transfer.py
# display final results and all-task comparison
```

Data is never committed. Students download from Kaggle once and place under `data/`.

---

## Development Workflow

```
1. implement src/ file
2. run local tests (CPU, small data): python src/.../file_test.py
3. fix until tests pass
4. update notebook and run on Colab GPU
5. save outputs and submission CSV
```

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

Imbalance ratio: 2.76x. Not extreme but enough to hurt macro F1 if ignored.
Mitigation: weighted CrossEntropy loss + WeightedRandomSampler + stratified split.

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

### Foundation (unblocks everything else)
- [ ] `src/config.py`
- [ ] `src/datasets/dataset.py` + `dataset_test.py`
- [ ] `src/evaluation/metrics.py` + `submission.py`
- [ ] `src/training/train.py` + `early_stopping.py` + `train_test.py`
- [ ] `src/evaluation/plots.py`

### Task 1
- [ ] `src/datasets/eda.py` + `eda_plots.py`
- [ ] `src/models/mlp.py` + `models_test.py`
- [ ] `task1_mlp.py`

### Task 2
- [ ] `src/models/cnn.py` (extend `models_test.py`)
- [ ] `task2_cnn.py`

### Task 3
- [ ] `src/models/transfer.py` (extend `models_test.py`)
- [ ] `task3_transfer.py`

### Final
- [ ] `notebook.ipynb`
- [ ] Run all tasks end-to-end on Colab, validate all submission CSVs
