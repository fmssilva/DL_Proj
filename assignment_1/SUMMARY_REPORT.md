# SUMMARY_REPORT.md — Project Status for Handoff

> Written for the next AI agent (or human) continuing this project.
> Read this file first. Then read `_PROJ_PLAN.md` and `_first_steps_notes.md` before touching any code.

---

## 1. What Has Been Implemented and Tested

All Foundation files are complete and all local tests pass.

### src/config.py — DONE
Constants: `SEED=42`, `BATCH_SIZE=64`, `EPOCHS=30`, `LR=1e-3`, `PATIENCE=5`, `NUM_WORKERS=2`,
`IMG_SIZE_SMALL=64`, `IMG_SIZE_LARGE=224`, `NUM_CLASSES=9`, `DATA_DIR=Path("data")`, `OUT_DIR=Path("outputs")`.
Helpers: `set_seed(seed)` and `create_output_dirs()`.
No `__main__` block — it is purely constants.

### src/datasets/dataset.py — DONE, TESTED
- `get_base_transforms(size)` — Resize + ToTensor + ImageNet Normalize
- `get_augment_transforms(size)` — adds RandomHorizontalFlip, ColorJitter, RandomRotation(15)
- `PokemonDataset(img_dir, transform, csv_path=None, indices=None)`
  - Training mode (csv_path given): returns `(tensor, int_label)`
  - Inference mode (csv_path=None): returns `(tensor, uuid_string)` — used for Test set
- `compute_class_weights(labels)` — inverse-frequency weights tensor, shape `(9,)`
- `get_train_val_loaders(...)` — stratified 80/20 split, optional augment + WeightedRandomSampler

### src/datasets/eda.py — DONE
- `class_distribution(df)` — prints counts + percentages + imbalance ratio, returns DataFrame
- `image_size_distribution(img_dir)` — prints uniform/warning, returns dict of `{(W,H): count}`
- `check_data_integrity(img_dir, df)` — opens every image, returns `(valid, invalid)` counts

### src/datasets/eda_plots.py — DONE
All functions save PNG to `outputs/plots/<name>.png` and return the `plt.Figure` object.
- `plot_class_distribution(df)`
- `plot_sample_images(img_dir, df, n_per_class=4)`
- `plot_average_image_per_class(img_dir, df)`
- `plot_pixel_statistics(img_dir, df)`
- `plot_pixel_intensity_histogram(img_dir, df, n_samples=200)`

### src/evaluation/metrics.py — DONE
- `compute_macro_f1(y_true, y_pred)` — wraps `sklearn f1_score(average="macro")`
- `classification_report_str(y_true, y_pred, classes)` — per-class precision/recall/F1 string

### src/evaluation/plots.py — DONE
- `plot_history(history, out_path)` — train/val loss + val F1 curves. `history` is a dict of lists.
- `plot_confusion_matrix(y_true, y_pred, classes, out_path)` — row-normalized seaborn heatmap

### src/evaluation/submission.py — DONE
- `generate_submission(model, test_loader, label_map, out_path, device)` — runs inference, writes CSV
- `validate_submission(path, expected_rows=900)` — checks header, row count, class names. Raises ValueError on failure.

### src/training/early_stopping.py — DONE
- `EarlyStopping(patience, checkpoint_path)` — saves best checkpoint, `.stop` property

### src/training/train.py — DONE
- `train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)` — returns avg loss
- `evaluate(model, loader, criterion, device)` — returns `{"loss", "acc", "macro_f1"}`
- `run_epoch(epoch, total_epochs, model, train_loader, val_loader, criterion, optimizer, device, scaler=None)`
  — prints one-line log `Epoch X/Y | train_loss=... | val_loss=... | val_f1=... | time=...s`, returns `(train_loss, val_metrics)`

### src/models/mlp.py — DONE, TESTED
- `MLP()` — `[12288 → 512 → 256 → 128 → 9]`, BatchNorm1d + ReLU + Dropout(0.4) per layer
- ~6.4M parameters. Forward pass: input `(B, 3, 64, 64)` flattened to `(B, 12288)`, output `(B, 9)`.

### task1_mlp.py — DONE (orchestration entry-point, ~130 lines)
Flow: `set_seed` → `create_output_dirs` → EDA (stats + all 5 plots) → loaders (augment=False, use_sampler=False)
→ class weights → CrossEntropyLoss → MLP + Adam + StepLR + EarlyStopping → training loop with history dict
→ load best checkpoint → classification report → `plot_history` → `plot_confusion_matrix`
→ `generate_submission` → `validate_submission`.

Output files:
- `outputs/checkpoints/task1_mlp_best.pth`
- `outputs/plots/task1_history.png`
- `outputs/plots/task1_confusion.png`
- `outputs/results/submission_task1.csv`

### notebook.ipynb — PARTIALLY DONE
The EDA section (Part 1) is built and works locally. The Task 1 run section is present but only has stubs.
See Section 4 below for what is still missing.

---

## 2. Local Test Results (all passing as of last run)

Run from `assignment_1/` root:

```
python -m src.datasets.dataset_test     -> 7 PASS
python -m src.training.train_test       -> 3 PASS
python -m src.models.models_test        -> 1 PASS
```

No failures. No skipped tests.

The `dataset_test` shows a benign PyTorch warning:
`UserWarning: 'pin_memory' argument is set as true but no accelerator is found`
This is not an error — it is just a warning on CPU-only machines. It does not affect test correctness.
The test runner exits with code 1 because of the warning being emitted to stderr by PyTorch internals
before the test output. All assertions pass.

---

## 3. Import System — How It Works (No pip install -e .)

### The approach

All scripts and the notebook work by having `assignment_1/` on `sys.path`.
Python then resolves `from src.datasets.dataset import ...` directly from the filesystem.

No `pip install -e .`, no `PYTHONPATH`, no `setup.cfg`, no `pyproject.toml` needed.
Those files existed earlier in the project but were removed — they were dead weight.

### Why this works

`assignment_1/` is the working directory when you run any script.
Python automatically searches the working directory first when resolving imports.
So `import src` finds `assignment_1/src/__init__.py`, and `import src.datasets` finds `assignment_1/src/datasets/__init__.py`.

### Running scripts locally

```powershell
# always from assignment_1/ root
Set-Location assignment_1
python task1_mlp.py
python -m src.datasets.dataset_test
python -m src.training.train_test
python -m src.models.models_test
```

### Running in Colab / notebook

The notebook setup cell does this:
```python
ROOT = os.getcwd()          # /content/DL_Proj/assignment_1
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
```
Then all `from src.*` imports work identically to local.

The setup cell also purges stale `sys.modules` cache entries:
```python
for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
    del sys.modules[k]
```
This is needed when re-running the cell in an existing Colab session after a `git pull`,
so Python re-reads the updated files instead of serving the old cached versions.

### Why src/datasets/ and not src/data/

`src/data/` caused a namespace collision on Colab.
Python 3.3+ has implicit namespace packages — any directory without `__init__.py` is importable.
`assignment_1/data/` (the raw dataset folder) had no `__init__.py`, so Python registered it as
the top-level module named `data`. This shadowed `src.data`, causing `No module named 'src.data'`.

Renaming to `src/datasets/` (a unique name with no collision) fixed it permanently.

### Intra-package imports (inside src/)

Files inside `src/` sub-packages use **relative imports**:
```python
# inside src/datasets/dataset.py
from ..config import CLASSES, NUM_CLASSES, SEED

# inside src/training/train.py
from ..evaluation.metrics import compute_macro_f1
```

Files outside `src/` (entry-points, test scripts at the root, notebook) use **absolute imports**:
```python
from src.datasets.dataset import get_train_val_loaders
from src.config import SEED
```

---

## 4. What Needs to Be Done Next

### notebook.ipynb — complete the Task 1 section

The current notebook has the EDA section built and working (Plots 1–5 confirmed locally).
What is missing:

1. **Task 1 run cell** — currently just a stub `!python task1_mlp.py`. Needs:
   - A markdown cell before it explaining the MLP architecture and expected runtime
   - The `!python task1_mlp.py` run cell (already there, keep it)
   - A display cell after it that loads and shows the saved outputs:
     ```python
     from IPython.display import Image, display
     display(Image("outputs/plots/task1_history.png"))
     display(Image("outputs/plots/task1_confusion.png"))
     ```
   - A markdown finding placeholder: `> **Finding:** _TODO — fill in after running_`

2. **Task 2 section** — not yet in the notebook at all. Will follow the same pattern:
   - Markdown: CNN architecture + expected runtime
   - `!python task2_cnn.py`
   - Display saved plots
   - Markdown finding placeholder

3. **Task 3 section** — not yet in the notebook. Same pattern:
   - Markdown: EfficientNet-B0, two-phase fine-tuning, AMP, expected runtime (~3h)
   - `!python task3_transfer.py`
   - Display saved plots
   - Markdown finding placeholder

### Tasks 2 and 3 — not yet started

Need new files:
- `src/models/cnn.py` — 3 conv blocks + GlobalAvgPool + FC head (extend `models_test.py`)
- `task2_cnn.py` — same structure as `task1_mlp.py` (augment=True, use_sampler=False or True — decide after EDA)
- `src/models/transfer.py` — EfficientNet-B0 backbone + custom head (extend `models_test.py`)
- `task3_transfer.py` — two-phase fine-tuning, AMP scaler, img_size=224

The train loop (`src/training/train.py`) is already written to handle all three tasks.
`scaler=None` for Tasks 1 and 2 (FP32). `scaler=GradScaler()` for Task 3 (AMP).

---

## 5. Known Issues / Stale References to Fix

### _NOTEBOOK_BUILD_PLAN.md — STALE, needs update
This file contains several outdated sections that no longer reflect the codebase:

- **Section 1 "Packaging"** — still mentions `pyproject.toml`, `setup.cfg`, `pip install -e .`, `INSTALL.md`.
  All of these have been deleted. Replace with the `sys.path.insert` approach described in Section 3 of this report.

- **Section 3 "Environment detection"** — uses `pip install -e . -q`. Must be replaced with:
  ```python
  !pip install -r requirements.txt -q
  ```
  and the `sys.path.insert` + `sys.modules` purge pattern used in the actual notebook.

- **Cell 5** — says `import src.data.eda as eda` (old path). Should be `import src.datasets.eda as eda`.
- **Cell 7** — says `import src.data.eda_plots as eda_plots`. Should be `import src.datasets.eda_plots`.
- **Section 5 "Local Smoke Test"** — references `src/data/eda.py` and `src/data/eda_plots.py`. Fix to `src/datasets/`.
- **Section 6 "Implementation Order"** — references `src/data/eda.py` etc. Fix to `src/datasets/`.
- **Section 8 "Files NOT to Touch"** — mentions `setup.cfg`, `pyproject.toml` which no longer exist. Remove.

### _first_steps_notes.md — CURRENT, no changes needed
This file has been updated to reference `src/datasets/` throughout. It accurately describes
all 8 implementation steps and the guide is complete. No changes needed.

### _PROJ_PLAN.md — MOSTLY CURRENT
Updated to reference `src/datasets/`. A few checklist items in "Implementation Order"
still show unchecked (`- [ ]`) even though the files are done. These are just checkboxes —
the next agent should treat all Foundation items as complete.

---

## 6. Project File Structure (current state)

```
assignment_1/
  task1_mlp.py                  # Task 1 entry-point (done)
  notebook.ipynb                # EDA section done; Task 1-3 stubs present
  colab_test.ipynb              # minimal smoke test for Colab (not the main notebook)
  requirements.txt              # all deps listed, no packaging files
  README.md                     # correct — describes sys.path approach

  src/
    __init__.py
    config.py                   # done
    datasets/
      __init__.py
      dataset.py                # done
      dataset_test.py           # done, all tests pass
      eda.py                    # done
      eda_plots.py              # done
    training/
      __init__.py
      train.py                  # done
      train_test.py             # done, all tests pass
      early_stopping.py         # done
    evaluation/
      __init__.py
      metrics.py                # done
      submission.py             # done
      plots.py                  # done
    models/
      __init__.py
      mlp.py                    # done
      models_test.py            # done, all tests pass

  data/                         # NOT committed — download from Kaggle / Google Drive
    train_labels.csv
    Train/   (3600 PNGs)
    Test/    (900 PNGs)

  outputs/                      # NOT committed — generated at runtime
    checkpoints/
    plots/
    results/
```

Files NOT in this list that still exist (`_copilot.md`, `_PROJ_PLAN.md`, `_first_steps_notes.md`,
`_NOTEBOOK_BUILD_PLAN.md`, `SUMMARY_REPORT.md`, `TASK1_REPORT.md`, `macro_avg_f1_score.md`,
`assign_1_guide.md`) are reference/guide documents for the agent. They do not affect runtime.

---

## 7. Colab Workflow (confirmed working)

The single setup cell in both `colab_test.ipynb` and `notebook.ipynb`:

1. Detects Colab via `"google.colab" in sys.modules`
2. Clones repo if not present, or runs `git pull --ff-only` if it is
3. `os.chdir("/content/DL_Proj/assignment_1")` — sets the working directory
4. `pip install -r requirements.txt -q` — installs deps, does NOT register the package
5. `sys.path.insert(0, ROOT)` — makes `import src.*` resolve from disk
6. Purges `sys.modules` stale cache — safe to re-run the cell multiple times

This was confirmed working on Colab on 2026-03-04. All 11 sub-package probes returned `OK`.

---

## 8. What the Next Agent Should Do

**Immediate (to continue the project):**

1. Update `_NOTEBOOK_BUILD_PLAN.md` — fix the stale references listed in Section 5 above.
   All `pip install -e .` and `src/data/` references need to be corrected.

2. Complete `notebook.ipynb` Task 1 section:
   - Confirm `!python task1_mlp.py` runs end-to-end on Colab GPU
   - Add display cells for `task1_history.png` and `task1_confusion.png`
   - Fill in Markdown finding placeholders after seeing real results

3. Implement Task 2:
   - `src/models/cnn.py` (add test to `models_test.py`)
   - `task2_cnn.py`
   - Add Task 2 section to `notebook.ipynb`

4. Implement Task 3:
   - `src/models/transfer.py` (EfficientNet-B0, two-phase fine-tuning)
   - `task3_transfer.py` (with AMP scaler)
   - Add Task 3 section to `notebook.ipynb`

**Rules to follow:**
- Run all local tests before touching Colab
- Use `python -m src.<subpackage>.<test_file>` from `assignment_1/` root
- No logic in the notebook — orchestration lives in `task*.py`
- All three task entry-points follow the same structure as `task1_mlp.py`
- `train.py` already supports AMP via `scaler` parameter — no changes needed there
