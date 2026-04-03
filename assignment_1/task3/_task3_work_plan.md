# Task 3 — Transfer Learning and Fine-Tuning Work Plan

> Read fully before touching any code.
> Transfer Learning architectures and fine-tunning strategies for Pokémon type classification. Systematic exploration:
> architecture sweep (clean) → augmentation → regularisation → ablations → ensembles.

---

## 0. Context — What Already Exists (Don't Duplicate)

All shared infrastructure from Task 2 is reused verbatim:

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

**File to rewrite:** `src/models/cnn.py` — replace old LeNet5/CNN/MediumCNN with proper CNN architectures  
**File to extend:** `src/models/models_test.py` — add CNN forward pass tests  
**File to extend:** `src/datasets/dataset.py` — add `get_strong_aug_transforms` + `train_transform` param  
**The notebook:** `task2/task2.ipynb` — rewrite with CNN experiments, MLP reference runs, comparison

---

## 1. `src/models/cnn.py` — Architectures

### Design principles
- No softmax — `CrossEntropyLoss` handles it
- Default `dropout=0.5` (higher than MLP because GlobalAvgPool compresses to a small vector)
- `AdaptiveAvgPool2d(1)` (true GlobalAvgPool) replaces Flatten — far fewer params, less overfit
- Conv block pattern: `Conv2d(k=3, p=1) → BN → ReLU → MaxPool2d(2,2)`
- No Dropout inside conv blocks — only in the FC head after GlobalAvgPool
- Input: `(B, 3, 64, 64)`. Output: `(B, 9)` logits.

### Classes (6 total)

#### `BaseCNN` — 3-block baseline (~120K params)
```
Conv(3→32,  k=3, p=1) → BN → ReLU → MaxPool(2)  # 64→32
Conv(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)  # 32→16
Conv(64→128,k=3, p=1) → BN → ReLU → MaxPool(2)  # 16→8
GlobalAvgPool → [128] → Dropout(p) → Linear(128, 9)
```
Constructor: `BaseCNN(dropout=0.5, use_bn=True)`. Setting `use_bn=False` gives the NoBN ablation variant — no need for a separate class.

#### `DeepCNN` — 4 blocks, more depth
```
Conv(3→32)   → BN → ReLU → MaxPool  # 64→32
Conv(32→64)  → BN → ReLU → MaxPool  # 32→16
Conv(64→128) → BN → ReLU → MaxPool  # 16→8
Conv(128→256)→ BN → ReLU → MaxPool  # 8→4
GlobalAvgPool → [256] → Dropout(p) → Linear(256, 9)
```
Tests maximum useful depth on 64px input (4×4 after 4 pools is the limit).

#### `WideCNN` — 3 blocks, wider channels
```
Conv(3→64)   → BN → ReLU → MaxPool  # 64→32
Conv(64→128) → BN → ReLU → MaxPool  # 32→16
Conv(128→256)→ BN → ReLU → MaxPool  # 16→8
GlobalAvgPool → [256] → Dropout(p) → Linear(256, 9)
```
Same depth as BaseCNN but 2× wider — richer feature maps.

#### `SEBlock` — channel attention helper (used inside ResidualCNN)
```
squeeze:  GlobalAvgPool → [C]
excite:   FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid
scale:    multiply original feature map channel-wise
```
`reduction=8` keeps it lightweight.

#### `ResidualCNN` — 3 residual blocks + optional SE attention
Each residual block:
```
input → Conv(k=3,p=1) → BN → ReLU → Conv(k=3,p=1) → BN
      + 1×1 projection (only when in_ch ≠ out_ch)  ← skip connection
→ ReLU → [optional SEBlock] → MaxPool(2)
```
Full architecture:
```
ResBlock(3→64)   → MaxPool  # 64→32
ResBlock(64→128) → MaxPool  # 32→16
ResBlock(128→256)→ MaxPool  # 16→8
GlobalAvgPool → [256] → Dropout(p) → Linear(256, 9)
```
Constructor: `ResidualCNN(dropout=0.5, use_se=False)`.
`use_se=True` adds SE attention after each residual block.

#### `MultiScaleCNN` — parallel k=3 and k=5 branches per block
```
Block 1: [Conv(3→16,k=3,p=1) || Conv(3→16,k=5,p=2)] → cat→32ch → BN → ReLU → MaxPool
Block 2: [Conv(32→32,k=3,p=1)|| Conv(32→32,k=5,p=2)] → cat→64ch → BN → ReLU → MaxPool
Block 3: [Conv(64→64,k=3,p=1)|| Conv(64→64,k=5,p=2)] → cat→128ch → BN → ReLU → MaxPool
GlobalAvgPool → [128] → Dropout(p) → Linear(128, 9)
```

---

## 2. Extend `src/models/models_test.py`

Add CNN forward-pass tests after existing MLP tests. Run: `python -m src.models.models_test`

---

## 3. `src/datasets/dataset.py` — Two additions

### 3a. `get_strong_aug_transforms(size)`
Aggressive augmentation with RandomErasing (after ToTensor), stronger ColorJitter, RandomVerticalFlip, RandomAffine.

### 3b. `train_transform` parameter on `get_train_val_loaders`
When provided, overrides the augment/grayscale logic for the train loader. Val transform is never overridden.

---

## 4. Notebook Structure: `task2/task2.ipynb`

### Part 0 — Project Load (identical to task1)
- Config, imports, data download, EDA (skip t-SNE — already done in task1)

### Part 1 — MLP Reference (kept from task1 for comparison)
- Re-run best MLP solo (R_ls015_drop03 recipe: MLP, drop=0.3, LS=0.15, class weights)
- This gives us an in-notebook MLP baseline number to compare all CNN experiments against

### Part 2 — CNN Experiments (systematic)

**Phase 1 — Architecture sweep (no augmentation, CE+weights)**

| ID  | Name            | Architecture          | What it tests                 |
| --- | --------------- | --------------------- | ----------------------------- |
| A   | `A_base_cnn`    | BaseCNN, drop=0.5     | CNN baseline — beat MLP?      |
| B   | `B_wide`        | WideCNN, drop=0.5     | Width: more filters per layer |
| C   | `C_deep`        | DeepCNN, drop=0.5     | Depth: 4 blocks               |
| D   | `D_residual`    | ResidualCNN, se=False | Skip connections              |
| E   | `E_residual_se` | ResidualCNN, se=True  | SE channel attention          |
| F   | `F_multiscale`  | MultiScaleCNN         | Multi-scale receptive fields  |

→ Identify BEST_ARCH from leaderboard after F.

**Phase 2 — Augmentation (best architecture)**

| ID  | Name                | Aug      | What it tests           |
| --- | ------------------- | -------- | ----------------------- |
| G   | `G_best_aug`        | Standard | Standard augmentation   |
| H   | `H_best_strong_aug` | Strong   | Aggressive augmentation |

**Phase 3 — Regularisation & loss variants (best arch + aug)**

| ID  | Name             | Sampler | Loss               | What it tests            |
| --- | ---------------- | ------- | ------------------ | ------------------------ |
| I   | `I_best_sampler` | Yes     | CE (no weights)    | Sampler vs weighted loss |
| J   | `J_best_ls`      | No      | CE+weights, LS=0.1 | Label smoothing for CNN  |

**Phase 4 — Ablations (no augmentation, clean comparisons)**

| ID  | Name          | Architecture          | What it tests               |
| --- | ------------- | --------------------- | --------------------------- |
| K   | `K_drop03`    | BEST_ARCH, drop=0.3   | Dropout 0.3 vs 0.5          |
| L   | `L_no_bn`     | BaseCNN(use_bn=False) | Quantify BN contribution    |
| M   | `M_cosine_lr` | BEST_ARCH             | CosineAnnealingLR vs StepLR |

### Part 3 — Ensembles
- `ENS_top2`: top-2 CNN solos
- `ENS_cnn_mlp`: best CNN + best MLP (cross-task diversity test)

### Part 4 — Evaluation & Comparison
- Leaderboard + per-class F1 heatmap
- Best model deep dive (classification report, confusion matrix, training curves)
- MLP vs CNN comparison table (load task1 JSON)
- Final submission

---

## 5. Expected Results

- **A_base_cnn** should comfortably beat MLP (~0.24 → ~0.35+)
- **D_residual** expected biggest jump from skip connections
- **G augmentation delta**: +0.03 to +0.05 F1 (CNN benefits more from spatial aug than MLP)
- **L_no_bn**: expect BN contributes +0.04–0.08 F1
- **ENS_cnn_mlp**: MLP likely drags CNN down — valid negative finding
- **Target**: val_macro_f1 ≥ 0.45–0.50 before Task 3

---

## 6. What NOT to do

- Do not change `train.py`, `early_stopping.py`, `persistence.py`, `submission.py`, `ensemble.py`
- Do not use `Flatten` + large FC in CNN — `GlobalAvgPool` is the right head
- Do not add Dropout inside conv blocks — only in the FC head
- Do not run augmentation before the architecture sweep — confounds comparisons
- Do not use `IMG_SIZE=128` — 1h budget doesn't allow it
- Do not add a grayscale section — colour is valuable for CNN
- Do not apply `RandomErasing` before `ToTensor`
