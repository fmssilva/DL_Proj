# NOTEBOOK_BUILD_PLAN.md

> Agent instructions for building `notebook.ipynb`.
> Read this file fully before writing a single line.
> Also read `_PROJ_PLAN.md` and all existing `src/` files before implementing.

---

## 1. Project Status — What Is Already Built and Tested

All foundation code lives in `src/`. Every file has been implemented and tested locally.
The `pip install -e .` editable install is in place — all `from src.*` imports work with no `sys.path` hacks.

### src/config.py
Constants: `SEED=42`, `BATCH_SIZE=64`, `EPOCHS=30`, `LR=1e-3`, `PATIENCE=5`, `NUM_WORKERS=2`,
`IMG_SIZE_SMALL=64`, `IMG_SIZE_LARGE=224`, `NUM_CLASSES=9`, `CLASSES` list (alphabetical).
Paths: `DATA_DIR=Path("data")`, `OUT_DIR=Path("outputs")`.
Helpers: `set_seed(seed)`, `create_output_dirs()`.

### src/datasets/dataset.py
- `get_base_transforms(size)` — Resize + ToTensor + ImageNet Normalize
- `get_augment_transforms(size)` — adds RandomHorizontalFlip, ColorJitter, RandomRotation(15)
- `PokemonDataset(img_dir, transform, csv_path=None, indices=None)`
  - Training mode (csv_path given): returns `(tensor, int_label)`
  - Inference mode (csv_path=None): returns `(tensor, uuid_string)` — used for test set
- `compute_class_weights(labels)` — inverse-frequency weights tensor, shape `(9,)`
- `get_train_val_loaders(csv_path, img_dir, img_size, batch_size, augment, use_sampler, num_workers)` — stratified 80/20 split

### src/datasets/eda.py  (stats, no plots, prints to stdout)
- `class_distribution(df)` — counts + percentages + imbalance ratio, prints summary, returns DataFrame
- `image_size_distribution(img_dir)` — dict of `{(W,H): count}`, prints uniform/warning
- `check_data_integrity(img_dir, df=None)` — tries to open every image, prints bad filenames, returns `(valid, invalid)` counts

### src/datasets/eda_plots.py  (figures, no stdout stats)
All functions: save PNG to `outputs/plots/<filename>.png` AND return the `plt.Figure` object.

| Function                                                     | Saved file                           | What it shows                                                  |
| ------------------------------------------------------------ | ------------------------------------ | -------------------------------------------------------------- |
| `plot_class_distribution(df)`                                | `plot_class_distribution.png`        | Horizontal bar chart, count + % per class, sorted descending   |
| `plot_sample_images(img_dir, df, n_per_class=4)`             | `plot_sample_images.png`             | Grid: n images per class, fixed seed                           |
| `plot_average_image_per_class(img_dir, df)`                  | `plot_average_image_per_class.png`   | Mean image per class (blurry = high variance)                  |
| `plot_pixel_statistics(img_dir, df)`                         | `plot_pixel_statistics.png`          | Per-channel R/G/B mean + std bar chart; prints computed values |
| `plot_pixel_intensity_histogram(img_dir, df, n_samples=200)` | `plot_pixel_intensity_histogram.png` | R/G/B overlay histogram on random sample                       |

### src/evaluation/metrics.py
- `compute_macro_f1(y_true, y_pred)` — macro F1, the competition metric
- `classification_report_str(y_true, y_pred, classes)` — per-class precision/recall/F1 string

### src/evaluation/plots.py  (figures, caller specifies out_path)
| Function                                                   | Saved by task scripts               | What it shows                              |
| ---------------------------------------------------------- | ----------------------------------- | ------------------------------------------ |
| `plot_history(history, out_path)`                          | `outputs/plots/task1_history.png`   | Train/val loss + val F1 curves over epochs |
| `plot_confusion_matrix(y_true, y_pred, classes, out_path)` | `outputs/plots/task1_confusion.png` | Row-normalised seaborn heatmap             |

`history` dict keys: `train_loss`, `val_loss`, `train_f1` (NaN for MLP), `val_f1`.

### src/evaluation/submission.py
- `generate_submission(model, test_loader, label_map, out_path, device)` — runs inference, writes CSV
- `validate_submission(path, expected_rows=900)` — checks header, row count, class names

### src/training/early_stopping.py
- `EarlyStopping(patience, checkpoint_path)` — saves best checkpoint, `.stop` property

### src/training/train.py
- `train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)` — returns avg loss
- `evaluate(model, loader, criterion, device)` — returns `{"loss", "acc", "macro_f1"}`
- `run_epoch(epoch, total_epochs, model, train_loader, val_loader, criterion, optimizer, device, scaler=None)` — prints one-line log, returns `(train_loss, val_metrics)`

### src/models/mlp.py
- `MLP()` — `[12288 → 512 → 256 → 128 → 9]`, BatchNorm+ReLU+Dropout(0.4) per layer, ~6.4M params

### task1_mlp.py  (entry-point orchestrator, ~130 lines)
Flow: `set_seed` → `create_output_dirs` → EDA (stats + all 5 plots) → loaders → class weights →
MLP → train loop → load best checkpoint → `classification_report_str` → `plot_history` →
`plot_confusion_matrix` → `generate_submission` → `validate_submission`.
Saves to: `outputs/checkpoints/task1_mlp_best.pth`, `outputs/plots/task1_history.png`,
`outputs/plots/task1_confusion.png`, `outputs/results/submission_task1.csv`.

### Packaging
- `pyproject.toml` + `setup.cfg` at `assignment_1/` root — `pip install -e .` makes all imports work
- `requirements.txt` — all dependencies listed
- `INSTALL.md` — instructions for local, Colab, Kaggle

### Data (NOT committed — download from Kaggle)
```
data/
  train_labels.csv   # 3600 rows: Id (UUID), label (string class name)
  Train/             # 3600 PNG images
  Test/              # 900 PNG images
```
Known numbers: 3600 train, 900 test, 9 classes, imbalance ratio 2.76x (Water 674 vs Ground 244).

---

## 2. Architecture Decision — Notebook Role

The notebook is a **thin display layer**. All logic lives in `src/`. All orchestration lives in `task*.py`.

**What belongs in the notebook:**
- Environment setup (clone repo, install package, download data prompt)
- EDA: call `eda.*` and `eda_plots.*` directly — each call is one line, no logic, not duplicated
- Task runs: `!python task1_mlp.py`, `!python task2_cnn.py`, `!python task3_transfer.py`
- Display: load and show saved PNGs from `outputs/plots/` after each script finishes

**What does NOT belong in the notebook:**
- Training loops (`for epoch in range(...)`)
- Model construction (`model = MLP().to(device)`)
- Loss / optimizer / scheduler setup
- Prediction collection loops
- Any import from `src.training`, `src.models`, `src.evaluation`

**Why this split:**
- No logic is duplicated between `task*.py` and the notebook
- Each `task*.py` is self-contained and runnable standalone (`python task1_mlp.py`)
- The notebook stays readable top-to-bottom with no dense training code blocks
- Tasks 2 and 3 follow exactly the same pattern — consistency across all three tasks
- Students can run any task from the terminal without opening the notebook

**Why the EDA calls are fine directly in the notebook:**
Each EDA call is a single display line (`fig = eda_plots.plot_class_distribution(df)`) with no
logic attached. They are not "duplicated" in any meaningful sense — `task1_mlp.py` re-runs EDA
for its own printed record, while the notebook shows EDA inline before training for context.
EDA is fast (a few seconds) so the re-run cost is negligible.

---

## 3. Notebook Strategy

### Pattern per section
```
[Markdown]  ## Section title + 1-2 sentence context
[Code]      call src function or display saved PNG
[Markdown]  > **Finding:** _TODO — fill in after running_
```

### Display pattern for live EDA plots (called directly in notebook)
```python
fig = eda_plots.plot_class_distribution(df)
plt.show()
plt.close(fig)
```
Use `plt.show()` for live figures — simple, familiar to students.
`plt.close(fig)` after show frees memory and prevents figure accumulation.

### Display pattern for saved plots (produced by task scripts)
After `!python task1_mlp.py` finishes, the plots are saved. Load them with:
```python
from IPython.display import Image, display
display(Image("outputs/plots/task1_history.png"))
display(Image("outputs/plots/task1_confusion.png"))
```
Use `IPython.display.Image` for already-saved files — do not call `plt.show()` on them.

### Error visibility for script cells
Use `!python task1_mlp.py` directly — do not wrap in `subprocess` or `try/except` in the notebook.
If the script fails, the full traceback appears in the cell output. Wrapping hides errors.

### matplotlib backend note
Never set `matplotlib.use("Agg")` in the notebook. That backend switch is ONLY for headless
scripts like `quick_eda_test.py` where no display is available. In the notebook, the default
interactive backend renders plots inline automatically.

### Environment detection (Cell 1 — runs identically locally and in Colab)
```python
import os

if not os.path.exists("src"):
    # running in Colab — clone and install
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/fmssilva/DL_Proj.git"], check=True)
    os.chdir("DL_Proj/assignment_1")
    subprocess.run(["pip", "install", "-e", ".", "-q"], check=True)
    print("IMPORTANT: Download data from Kaggle and place under data/Train, data/Test, data/train_labels.csv")
else:
    print("Running locally — src/ found, skipping clone.")
```
This makes the same notebook file work locally AND in Colab without any edits.

---

## 4. Notebook File: `notebook.ipynb` — Cell Structure

**Cell 1 — Markdown: Title**
- Title + one-line dataset/task description

**Cell 2 — Markdown: Cell 1 — Environment Setup**

**Cell 3 — Environment Setup** (python)
- Environment detection as shown in Section 3
- `import torch, pandas as pd, matplotlib.pyplot as plt, from pathlib import Path`
- `from src.config import set_seed, create_output_dirs, DATA_DIR, OUT_DIR, SEED, CLASSES`
- `set_seed(SEED)`, `create_output_dirs()`
- `CSV_PATH`, `TRAIN_DIR`, `TEST_DIR` path variables
- `df = pd.read_csv(CSV_PATH)`
- Print device

**Cell 4 — Markdown: Part 1 — Exploratory Data Analysis**
- Section header + 1-2 sentences

**Cell 5 — EDA Stats** (python)
- `import src.data.eda as eda`
- `eda.class_distribution(df)`, `eda.image_size_distribution(TRAIN_DIR)`, `eda.check_data_integrity(TRAIN_DIR, df)`

**Cell 6 — Markdown: Plot 1 — Class Distribution**

**Cell 7 — Class Distribution Plot** (python)
- `import src.data.eda_plots as eda_plots`
- `fig = eda_plots.plot_class_distribution(df)` + `plt.show()` + `plt.close(fig)`

**Cell 8 — Markdown: Finding placeholder**

**Cell 9 — Markdown: Plot 2 — Sample Images per Class**

**Cell 10 — Sample Images Plot** (python)
- `fig = eda_plots.plot_sample_images(TRAIN_DIR, df, n_per_class=4)` + `plt.show()` + `plt.close(fig)`

**Cell 11 — Markdown: Finding placeholder**

**Cell 12 — Markdown: Plot 3 — Average Image per Class**

**Cell 13 — Average Image Plot** (python)
- `fig = eda_plots.plot_average_image_per_class(TRAIN_DIR, df)` + `plt.show()` + `plt.close(fig)`

**Cell 14 — Markdown: Finding placeholder**

**Cell 15 — Markdown: Plot 4 — Per-Channel Pixel Statistics**

**Cell 16 — Pixel Statistics Plot** (python)
- `fig = eda_plots.plot_pixel_statistics(TRAIN_DIR, df)` + `plt.show()` + `plt.close(fig)`

**Cell 17 — Markdown: Finding placeholder**

**Cell 18 — Markdown: Plot 5 — Pixel Intensity Histogram**

**Cell 19 — Pixel Intensity Histogram Plot** (python)
- `fig = eda_plots.plot_pixel_intensity_histogram(TRAIN_DIR, df, n_samples=200)` + `plt.show()` + `plt.close(fig)`

**Cell 20 — Markdown: Finding placeholder**

**Cell 21 — Markdown: Task 1 — MLP Baseline**
- Architecture description, training strategy, expected runtime (~20 min on Colab T4)

**Cell 22 — Task 1: Run** (python)
```python
!python task1_mlp.py
```
No wrapper. If the script fails, the traceback is visible here.

**Cell 23 — Task 1: Display Outputs** (python)
```python
from IPython.display import Image, display
display(Image("outputs/plots/task1_history.png"))
display(Image("outputs/plots/task1_confusion.png"))
```

**Cell 24 — Markdown: Finding placeholder**

---

## 5. Local Smoke Test

Before building the full notebook, run the `__main__` blocks in the EDA modules directly:
```
python src/datasets/eda.py
python src/datasets/eda_plots.py
```

`eda.py __main__` tests all 3 stats functions on the full dataset.
`eda_plots.py __main__` tests all 5 plot functions on a 50-image subset with `matplotlib.use("Agg")`.
Both must pass before the notebook EDA cells are considered working.

Note: `matplotlib.use("Agg")` is set inside `eda_plots.py __main__` for headless terminal runs.
Never set it in the notebook — the default interactive backend renders plots inline automatically.

---

## 6. Implementation Order for the Agent

```
1. Run smoke tests: python src/datasets/eda.py ; python src/datasets/eda_plots.py
2. Run all other tests: python src/datasets/dataset_test.py ; python src/training/train_test.py ; python src/models/models_test.py
3. Build notebook.ipynb with the cell structure above (Section 4)
4. Run notebook locally cell by cell — confirm each cell outputs something sensible
5. Fill in Markdown finding placeholders after seeing real results
6. Push to GitHub
7. Open in Colab — run setup cell, download data, run all cells
```

---

## 7. Key Constraints

- No training logic in the notebook — training lives in `task*.py` which calls `src/`
- EDA plot calls in the notebook are acceptable (one-liners, no duplicated logic)
- `plt.show()` + `plt.close(fig)` after every live plot call
- `IPython.display.Image` for displaying already-saved PNGs from task scripts
- Markdown finding placeholders: `> **Finding:** _TODO — fill in after running_`
- Environment detection in Cell 3 must work with zero edits between local and Colab
- No emojis anywhere (terminal encoding issues)
- Script cells use `!python task*.py` directly — no subprocess wrapper, no try/except
- The notebook file is `assignment_1/notebook.ipynb` (not in a subfolder)

---

## 8. Files NOT to Touch

Everything in `src/` is done and tested. Do not modify any `src/` file.
Do not modify `task1_mlp.py`, `requirements.txt`, `setup.cfg`, `pyproject.toml`.
Only create/edit: `notebook.ipynb`.

