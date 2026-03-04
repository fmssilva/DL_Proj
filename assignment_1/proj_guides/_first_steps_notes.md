# Initial Steps Guide — Foundation + Task 1 (EDA + MLP)

> Read `_PROJ_PLAN.md` before writing any code.
> This guide is a step-by-step implementation task list.
> Implement in order — each step unblocks the next.

---

## Why this order

```
Step 1: config.py                               <- constants + set_seed + output dirs
Step 2: dataset.py + dataset_test.py            <- data pipeline, everything else needs this
Step 3: metrics.py + submission.py              <- train.py imports metrics.py, so this comes first
Step 4: train.py + early_stopping.py + train_test.py
Step 5: eda.py + eda_plots.py                   <- needs dataset.py and config.py
Step 6: evaluation/plots.py                     <- training curves + confusion matrix
Step 7: mlp.py + models_test.py
Step 8: task1/notebook.ipynb                    <- wires everything together, IS the main()
```

The foundation files (config, dataset, metrics, train loop) are shared by all 3 tasks.
Getting them right first means Tasks 2 and 3 only need to add a model and notebook cells.
`metrics.py` comes before `train.py` because `train.py` imports `compute_macro_f1` from it.

---

## Step 1 — src/config.py ✓ DONE

Single source of truth for all constants.

What is implemented:
- `SEED`, `FAST_RUN`, `BATCH_SIZE`, `EPOCHS`, `LR`, `PATIENCE`, `NUM_WORKERS`
- `FAST_RUN = True` sets `EPOCHS=4`, `PATIENCE=2` for quick pipeline testing. Set `False` for real runs.
- `IMG_SIZE_SMALL = 64` (MLP + CNN), `IMG_SIZE_LARGE = 224` (Transfer)
- `NUM_CLASSES = 9`
- `CLASSES` list (alphabetical: Bug, Fighting, Fire, Grass, Ground, Normal, Poison, Rock, Water)
- `DATA_DIR = Path("data")`, `OUT_DIR = Path("outputs")`
- `set_seed(seed)` — sets random, numpy, torch seeds + cudnn deterministic
- `create_output_dirs()` — creates outputs/checkpoints, outputs/results, outputs/plots

No `__main__` block — purely constants and helpers.

---

## Step 2 — src/datasets/dataset.py + dataset_test.py ✓ DONE

Core data pipeline. Everything else depends on this.

What is implemented in `dataset.py`:
- `get_base_transforms(size)` — Resize + ToTensor + Normalize (ImageNet mean/std)
- `get_augment_transforms(size)` — adds RandomHorizontalFlip, ColorJitter, RandomRotation(15)
- `PokemonDataset(img_dir, transform, csv_path=None, indices=None)`
  - Training mode (csv_path given): returns `(tensor, int_label)`
  - Inference mode (csv_path=None): lists all PNGs in `img_dir`, returns `(tensor, uuid_string)` — used for Test set
- `compute_class_weights(labels)` — inverse-frequency weights tensor, shape `(9,)`
- `get_train_val_loaders(csv_path, img_dir, img_size, batch_size, augment, use_sampler, num_workers)` — stratified 80/20 split

**Note on `src/datasets/` naming:** `src/data/` caused a namespace collision with `data/` (the raw dataset folder) on Colab. Python's implicit namespace packages registered `data/` as the top-level `data` module, shadowing `src.data`. Renamed to `src/datasets/` — unique name, no collision.

**Note on `use_sampler`:** Task 1 (MLP) uses `use_sampler=False` — weighted loss alone is sufficient for 2.76× imbalance. Revisit for Tasks 2 and 3.

What `dataset_test.py` checks:
- Dataset length matches CSV row count (3600)
- Image tensor shape `(3, H, W)`, dtype float32, values roughly in `[-3, 3]` after normalization
- Labels are integers in `[0, 8]`
- Stratified split preserves class ratios
- Class weights: shape `(9,)`, no zeros, Ground/Rock highest
- Inference mode: returns `(tensor, uuid_string)`, length 900

Run with: `python -m src.datasets.dataset_test`

---

## Step 3 — src/evaluation/metrics.py + submission.py ✓ DONE

What is implemented in `metrics.py`:
- `compute_macro_f1(y_true, y_pred)` — wraps `f1_score(average="macro")`
- `classification_report_str(y_true, y_pred, classes)` — per-class precision/recall/F1 string

What is implemented in `submission.py`:
- `generate_submission(model, test_loader, label_map, out_path, device)` — runs inference, decodes integer predictions to class name strings using `label_map[i]`, writes CSV with header `Id,label`
- `validate_submission(path, expected_rows=900)` — checks row count, header, class names. Raises `ValueError` with a clear message if anything is wrong.

---

## Step 4 — src/training/early_stopping.py + train.py + train_test.py ✓ DONE

What is implemented in `early_stopping.py`:
- `EarlyStopping(patience, checkpoint_path)` — saves best model, counts epochs without improvement
- `.stop` property — returns `True` when patience is exceeded

What is implemented in `train.py` (two functions only — no wrappers):
- `train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)` → float (avg loss)
  - `scaler != None` enables AMP (Task 3 only); `None` = FP32 (Tasks 1 & 2)
- `evaluate(model, loader, criterion, device)` → `{"loss", "acc", "macro_f1"}`

Per-epoch logging lives in the notebook cells directly so students see it. `train.py` is pure computation only.

What `train_test.py` checks:
- Loss decreases after 10 training steps on a tiny batch
- `evaluate()` returns dict with correct keys and values in expected ranges

Run with: `python -m src.training.train_test`

---

## Step 5 — src/datasets/eda.py + eda_plots.py ✓ DONE

What is implemented in `eda.py` (stats, no plots):
- `class_distribution(df)` — counts + percentage per class, imbalance ratio
- `image_size_distribution(img_dir)` — checks all image sizes, returns `{(W,H): count}`
- `check_data_integrity(img_dir, df)` — opens every image, returns `(valid, invalid)` counts

What is implemented in `eda_plots.py` (figures, no stdout stats):
- All functions save PNG to `outputs/plots/<function_name>.png` and return the `plt.Figure`
- `plot_class_distribution(df)`, `plot_sample_images(img_dir, df, n_per_class=4)`
- `plot_average_image_per_class(img_dir, df)`, `plot_pixel_statistics(img_dir, df)`
- `plot_pixel_intensity_histogram(img_dir, df, n_samples=200)`

---

## Step 6 — src/evaluation/plots.py ✓ DONE

What is implemented:
- `plot_history(history, out_path)` — train/val loss + val F1 curves. `history` is `{"train_loss", "val_loss", "val_f1"}` (lists of floats). Saves + returns fig.
- `plot_confusion_matrix(y_true, y_pred, classes, out_path)` — row-normalized seaborn heatmap. Saves + returns fig.

---

## Step 7 — src/models/mlp.py + models_test.py ✓ DONE

Architecture: `Flatten → FC(12288→512) → BN → ReLU → Dropout(0.4) → FC(512→256) → BN → ReLU → Dropout(0.4) → FC(256→128) → BN → ReLU → Dropout(0.4) → FC(128→9)`

- ~6.4M parameters. Output: logits for 9 classes (no softmax — CrossEntropyLoss handles it).

`models_test.py` checks: forward pass with `torch.randn(4, 3, 64, 64)` → output shape `(4, 9)`, no NaN, prints param count.

Run with: `python -m src.models.models_test`

---

## Step 8 — task1/notebook.ipynb ✓ DONE

The notebook IS the main orchestrator. No separate `task1_mlp.py` exists — all logic is inline in cells so students can read and understand every step.

Notebook cell structure:
1. **Setup** — env detect, git clone/pull on Colab, `%pip install`, `sys.path.insert`, shared imports
2. **Data** — Google Drive download if not present (no Kaggle account needed), load CSV
3. **EDA stats** — `eda.class_distribution`, `eda.image_size_distribution`, `eda.check_data_integrity`
4–8. **EDA plots** — one cell per plot, each calls an `eda_plots.*` function and calls `plt.show()`
9. **Data loaders + model setup** — `get_train_val_loaders`, `PokemonDataset` test loader, `MLP`, `CrossEntropyLoss(weight=...)`, Adam, StepLR, EarlyStopping
10. **Training loop** — explicit `for epoch in range(...)` with inline timing and print per epoch
11. **Evaluation** — load best checkpoint, `evaluate()`, `classification_report_str`
12. **Plots** — `plot_history`, `plot_confusion_matrix` displayed inline
13. **Submission** — `generate_submission`, `validate_submission`

---

## Libraries

All already in `requirements.txt`:
- `torch`, `torchvision` — model, transforms, DataLoader
- `scikit-learn` — train_test_split, f1_score, classification_report
- `pandas` — CSV loading
- `numpy` — pixel stat computation
- `Pillow` — image loading in EDA
- `matplotlib` + `seaborn` — all plots

---

## Known Data Facts (for test assertions)

```
Total train images:  3600
Total test images:   900
Classes:             9
Imbalance ratio:     2.76x (Water 674 vs Ground 244)

Class counts:
  Water:    674  (18.7%)
  Normal:   606  (16.8%)
  Poison:   467  (13.0%)
  Fire:     381  (10.6%)
  Bug:      374  (10.4%)
  Grass:    299   (8.3%)
  Fighting: 291   (8.1%)
  Rock:     264   (7.3%)
  Ground:   244   (6.8%)

ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
```

---

## Running Local Tests

Always from `assignment_1/` root:

```powershell
python -m src.datasets.dataset_test     # 7 PASS
python -m src.training.train_test       # 3 PASS
python -m src.models.models_test        # 1 PASS
```

The `dataset_test` emits a benign PyTorch warning about `pin_memory` on CPU-only machines. All assertions pass — this is not a failure.


