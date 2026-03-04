# Initial Steps Guide — Foundation + Task 1 (EDA + MLP)

> Read assignment_1/PROJ_PLAN.md and copilot.md before writing any code.
> Follow all agent guidelines there. This guide is a step-by-step task list.
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
Step 8: task1_mlp.py                            <- wires everything together
```

The foundation files (config, dataset, metrics, train loop) are shared by all 3 tasks.
Getting them right first means Tasks 2 and 3 only need to add a model and an entry-point.
metrics.py comes before train.py because train.py imports compute_macro_f1 from it.

---

## Step 1 — src/config.py

Single source of truth for all constants. Every other file imports from here.

What to implement:
- SEED, BATCH_SIZE, EPOCHS, LR, PATIENCE, NUM_WORKERS
- IMG_SIZE_SMALL = 64 (MLP + CNN), IMG_SIZE_LARGE = 224 (Transfer)
- NUM_CLASSES = 9
- CLASSES list (alphabetical: Bug, Fighting, Fire, Grass, Ground, Normal, Poison, Rock, Water)
- DATA_DIR = Path("data"), OUT_DIR = Path("outputs")
- set_seed(seed) — sets random, numpy, and torch seeds in one call. Call this at the top of every task entry-point.
- create_output_dirs() — calls OUT_DIR/checkpoints, OUT_DIR/results, OUT_DIR/plots .mkdir(parents=True, exist_ok=True). Call this once at the start of each task entry-point.

No __main__ block needed. No other functions — just constants and these two small helpers.

---

## Step 2 — src/datasets/dataset.py + dataset_test.py

Core data pipeline. Everything else depends on this.

What to implement in dataset.py:
- get_base_transforms(size) — Resize + ToTensor + Normalize with ImageNet mean/std
- get_augment_transforms(size) — adds RandomHorizontalFlip, ColorJitter(brightness, contrast, saturation), RandomRotation(15)
- PokemonDataset(csv_path, img_dir, transform) — reads train_labels.csv, maps UUID to file path, encodes labels as integers using sorted(CLASSES)
- compute_class_weights(labels) — returns a tensor of weights using total / (NUM_CLASSES * class_count), one weight per class
- get_train_val_loaders(csv_path, img_dir, img_size, batch_size, augment, use_sampler) — stratified 80/20 split, builds DataLoaders, optionally uses WeightedRandomSampler

PokemonDataset inference mode: when csv_path=None, the dataset is in inference mode — it lists all PNG files
in img_dir directly (no labels), and returns (image_tensor, uuid_stem) per item so the UUID can be written
to the submission CSV. The test loader for Task 1 uses this mode pointing at data/Test/.

Note on use_sampler: WeightedRandomSampler oversamples minority classes at the DataLoader level.
For Task 1 (MLP), use use_sampler=False — weighted loss alone is sufficient for a 2.76x imbalance ratio,
and the sampler adds complexity without meaningful gain at this stage. For Tasks 2 and 3, revisit this.
Document this decision in a comment in task1_mlp.py.

What to implement in dataset_test.py (__main__ block):
- Dataset length matches CSV row count (3600)
- Image tensor shape is (3, H, W), dtype float32, values in roughly [-3, 3] after normalization
- Labels are integers in [0, 8]
- Stratified split preserves class ratios — check with Counter on both splits
- Class weights: shape (9,), no zeros, Ground/Rock should have highest weights
- Augment transform output has same shape as base transform output
- Inference mode (csv_path=None): dataset over Test/ returns (tensor, uuid_string) tuples, no label, length 900

Run with: python src/datasets/dataset_test.py
Must pass fully on CPU before moving on.

---

## Step 3 — src/evaluation/metrics.py + submission.py

Evaluation utilities. Comes before train.py because train.py imports compute_macro_f1 from metrics.py.

What to implement in metrics.py:
- compute_macro_f1(y_true, y_pred) -> float — wraps sklearn f1_score(average="macro")
- classification_report_str(y_true, y_pred, classes) -> str — wraps sklearn classification_report, returns string for logging

What to implement in submission.py:
- generate_submission(model, test_loader, label_map: list, out_path) — runs inference on test set, decodes integer predictions to class name strings using label_map[i], writes CSV with header Id,label. Pass CLASSES from config directly — index i maps to CLASSES[i], no separate encoder object needed.
- validate_submission(path, expected_rows=900) — checks row count, header, class names are all in CLASSES. Raises ValueError with a clear message if anything is wrong. Call this immediately after generating.

No __main__ block needed for metrics.py. Add a small __main__ in submission.py that builds a dummy CSV and asserts validate_submission passes on it.

---

## Step 4 — src/training/early_stopping.py + train.py + train_test.py

Shared training infrastructure. Used identically by all 3 tasks.

What to implement in early_stopping.py:
- EarlyStopping(patience, checkpoint_path) class
  - __call__(val_loss, model) — saves best model, counts epochs without improvement
  - .stop property — returns True when patience is exceeded

What to implement in train.py:
- train_one_epoch(model, loader, criterion, optimizer, device, scaler=None) -> float (avg loss)
  - scaler is the AMP GradScaler for Task 3; if None runs normal FP32
- evaluate(model, loader, criterion, device) -> dict with keys: loss, acc, macro_f1
  - imports compute_macro_f1 from src/evaluation/metrics.py

Per-epoch logging — print exactly one line per epoch (no nested tqdm, no extra prints):
  Epoch {e}/{EPOCHS} | train_loss={:.4f} | val_loss={:.4f} | val_f1={:.4f} | time={:.1f}s
  Use time.time() around the epoch block. This is the only training log needed.

What to implement in train_test.py (__main__ block):
- Build a tiny model (nn.Linear), tiny DataLoader (2 batches of 4 samples), run train_one_epoch for 10 steps
- Assert final loss < initial loss (10 steps gives enough signal to be reliable — 2 steps is too noisy)
- Assert evaluate() returns dict with correct keys and values in [0, 1]

Run with: python src/training/train_test.py

---

## Step 5 — src/datasets/eda.py + eda_plots.py

EDA drives architecture decisions — document what each finding implies.
Run after Step 2 is working (needs dataset.py and config.py).

What EDA should cover and why it matters:

1. Class distribution — counts + percentages + imbalance ratio
   drives: confirms loss weight values, whether WeightedRandomSampler is justified

2. Image size distribution — are all images the same size?
   drives: confirms resize is safe, justifies 64x64 choice

3. Per-channel pixel statistics — mean and std per R/G/B across all training images
   drives: decide whether to use ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
   or compute dataset-specific stats. If dataset stats differ significantly, use dataset stats.

4. Sample images per class — visual grid
   drives: understand visual similarity between classes (Bug vs Grass both green, Fighting vs Normal both humanoid)
   informs augmentation choices

5. Average image per class — mean image of all samples per class
   drives: high intra-class variance = harder task, more augmentation needed

6. Pixel intensity histogram — R/G/B overlay on a random sample
   drives: confirms RGB (not grayscale), detects blank/corrupted images, shows contrast range

7. Data integrity check — iterate all files, try to open each
   drives: confidence that all 3600 train + 900 test images are readable before training

What to implement in eda.py (stats, no plots):
- class_distribution(df) -> DataFrame — counts and percentage per class, imbalance ratio (max/min). Print summary.
- image_size_distribution(img_dir) -> dict — checks all image sizes, returns unique sizes found. Log if not all the same.
- check_data_integrity(img_dir, df) -> (int, int) — (valid, invalid) counts. Log any unreadable files by filename.

What to implement in eda_plots.py (figures, no stdout stats):
- plot_class_distribution(df) — horizontal bar chart sorted by count, count + % annotated. Save to outputs/plots/. Return fig.
- plot_sample_images(img_dir, df, n_per_class=4) — grid, fixed seed for reproducibility. Save + return fig.
- plot_average_image_per_class(img_dir, df) — compute mean image per class, display as grid. Save + return fig.
- plot_pixel_statistics(img_dir, df) — per-channel mean + std bar chart with error bars. Print computed values. Save + return fig.
- plot_pixel_intensity_histogram(img_dir, df, n_samples=200) — R/G/B overlay histogram on random sample. Save + return fig.

All plot functions: save figure to outputs/plots/<function_name>.png, then return the fig object (so the notebook can call display(fig) inline).

Add __main__ blocks:
- eda.py: run all stat functions on the real CSV, print output
- eda_plots.py: run all plot functions on a 50-image subset (not full dataset — full dataset runs only from task1_mlp.py), assert returned figures are not None

---

## Step 6 — src/evaluation/plots.py

Training history and confusion matrix plots. Called by all 3 task entry-points after training.

What to implement:
- plot_history(history, out_path) — loss and macro_f1 curves over epochs (train + val on same axes). history is a dict of lists: {"train_loss", "val_loss", "train_f1", "val_f1"}. Save to outputs/plots/. Return fig.
- plot_confusion_matrix(y_true, y_pred, classes, out_path) — seaborn heatmap, normalized by row. Save to outputs/plots/. Return fig.

Add a __main__ block that:
- Builds dummy history (lists of 5 floats) and calls plot_history, asserts the output file exists
- Builds dummy y_true/y_pred (9 classes, 50 samples) and calls plot_confusion_matrix, asserts the output file exists
These are pure path/import sanity checks — catches problems before Task 1 runs.

---

## Step 7 — src/models/mlp.py + models_test.py

First model. Keep it simple and explainable.

Architecture: flatten input -> FC stack [Input -> 512 -> 256 -> 128 -> 9]
- BatchNorm1d + ReLU + Dropout(0.4) between each layer
- Input size: 64 * 64 * 3 = 12288 (flat pixel vector)
- Output: logits for 9 classes (no softmax — CrossEntropyLoss handles it)

What to implement in mlp.py:
- MLP(nn.Module) class
- __init__ builds the Sequential stack
- forward(x) flattens x then passes through stack

What to implement in models_test.py (__main__ block):
- MLP forward pass: dummy input torch.randn(4, 3, 64, 64), expected output shape (4, 9)
- Assert no NaN in output
- Print parameter count

Run with: python src/models/models_test.py

---

## Step 8 — task1_mlp.py

Entry-point that orchestrates everything for Task 1. Should be short (~60 lines max).

Flow:
1. set_seed(SEED) and create_output_dirs() — both from config.py
2. run EDA: call eda.class_distribution, eda.check_data_integrity, all eda_plots functions on full dataset
3. build train/val loaders using get_train_val_loaders (augment=False, use_sampler=False for MLP — see Step 2 note)
4. build test loader using PokemonDataset(csv_path=None, img_dir=TEST_DIR, transform=get_base_transforms(IMG_SIZE_SMALL))
5. compute class weights, build weighted CrossEntropyLoss
6. build MLP model, Adam optimizer, StepLR scheduler, EarlyStopping
7. training loop: for each epoch call train_one_epoch + evaluate, accumulate train_loss/val_loss/val_f1 into a history dict, feed result into EarlyStopping; break if .stop
8. load best checkpoint, run evaluate on val set, print classification_report_str
9. call generate_submission + validate_submission, save to outputs/results/submission_task1.csv
10. call plot_history and plot_confusion_matrix from evaluation/plots.py

What NOT to put here: no model definition, no transform logic, no metric computation. Just wiring.

---

## Libraries

All already in requirements.txt:
- torch, torchvision — model, transforms, DataLoader
- scikit-learn — train_test_split, f1_score, classification_report
- pandas — CSV loading
- numpy — pixel stat computation
- Pillow — image loading in EDA
- matplotlib + seaborn — all plots
- tqdm — progress bars when iterating images

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

Use these numbers in test assertions to catch any data loading bugs early.

