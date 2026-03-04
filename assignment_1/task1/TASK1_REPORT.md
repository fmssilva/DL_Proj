# Task 1 — MLP Classification Report
> Status tracker + content bullets for the final presentation slides.
> Write full prose later. This is the "what to say" skeleton.

---

## Rubric Breakdown

| Section                     | Weight | Status                                                   |
| --------------------------- | ------ | -------------------------------------------------------- |
| Data Exploration & Analysis | 6%     | code done — need real outputs from Colab run             |
| Model Development           | 10%    | done — need to justify every choice explicitly in slides |
| Training Efficiency         | 5%     | done — need to record GPU time + actual epochs run       |
| Performance Evaluation      | 10%    | code done — need real numbers from Colab run             |
| Presentation Quality        | 3%     | TODO after Colab run                                     |
| Peer Review                 | 1%     | TODO after submission                                    |

---

## 1. Data Exploration & Analysis (6%)

### What we have — DONE
- `eda.class_distribution(df)` → counts + % per class + imbalance ratio printed
- `eda.image_size_distribution(img_dir)` → confirms all images are same size
- `eda.check_data_integrity(img_dir, df)` → opens every file, checks CSV↔disk match
- `eda_plots.plot_class_distribution(df)` → horizontal bar chart, sorted, count + % annotated
- `eda_plots.plot_sample_images(img_dir, df, n=4)` → 9×4 grid, fixed seed, reproducible
- `eda_plots.plot_average_image_per_class(img_dir, df)` → mean pixel image per class
- `eda_plots.plot_pixel_statistics(img_dir, df)` → per-channel mean/std + printed comparison vs ImageNet
- `eda_plots.plot_pixel_intensity_histogram(img_dir, df)` → R/G/B overlay histogram

### Known data facts
- 3600 train / 900 test, 9 classes, all images same size
- **Class counts:** Water 674 (18.7%) → Ground 244 (6.8%) — imbalance ratio **2.76×**
- Pixel stats close to ImageNet reference → justifies reusing ImageNet normalization constants
- Visually similar class pairs: Bug/Grass (both greenish), Fighting/Normal (both humanoid) → expected confusion matrix hotspots

### TODO before slides
- [ ] Run notebook on Colab, save all 5 EDA plots to `task1/outputs/plots/`
- [ ] Fill in "Finding" TODO cells in the notebook with actual observations
- [ ] Slide bullet: explain WHY macro-F1 not accuracy (model always predicting Water = 19% acc but ~0.02 macro-F1)
- [ ] State which class pairs are visually similar → sets up confusion matrix interpretation

---

## 2. Model Development (10%)

### Architecture — DONE
```
Input: 64×64×3 = 12,288 flat features
Flatten
→ Linear(12288, 512) → BatchNorm1d → ReLU → Dropout(0.4)
→ Linear(512, 256)   → BatchNorm1d → ReLU → Dropout(0.4)
→ Linear(256, 128)   → BatchNorm1d → ReLU → Dropout(0.4)
→ Linear(128, 9)     [logits, no softmax — CrossEntropyLoss handles that]
Total params: ~6.4M
```

### Justification table (use in slides)
| Choice                        | Why                                                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Flatten input                 | MLP has no spatial inductive bias — pixels treated as independent features                                         |
| 3 hidden layers 512→256→128   | Progressive compression; wider early = more feature combos; deeper = better abstraction                            |
| BatchNorm1d after each Linear | Reduces internal covariate shift → faster convergence, more stable (Ioffe & Szegedy 2015)                          |
| ReLU                          | No vanishing gradient for positive inputs; cheaper than sigmoid/tanh                                               |
| Dropout(0.4)                  | 12,288 input features = high co-adaptation risk; 0.4 balances regularization vs capacity                           |
| No softmax at output          | CrossEntropyLoss applies log_softmax internally; adding Softmax layer = redundant + numerically unstable           |
| Weighted CrossEntropyLoss     | `weight = total / (9 * class_count)` → Ground ≈1.64, Water ≈0.59; loss penalizes minority misclassifications more  |
| Adam lr=1e-3                  | Adaptive per-param LR; default from Kingma & Ba 2015; robust across tasks                                          |
| StepLR(step=5, γ=0.5)         | Halves LR every 5 epochs → prevents overshooting once loss flattens                                                |
| EarlyStopping(patience=5)     | Stops when val_loss stops improving → avoids overfitting tail, saves compute                                       |
| No augmentation               | Augmentation is spatial (flip, rotate); MLP destroys spatial info by flattening — augmentation is meaningless here |
| use_sampler=False             | 2.76× imbalance is mild (sampler adds value at >5×); weighted loss alone is sufficient                             |
| ImageNet normalization        | Pokémon pixel stats ≈ ImageNet (confirmed by plot_pixel_statistics)                                                |

### TODO
- [ ] Quick experiment: `label_smoothing=0.1` in CrossEntropyLoss — softens hard targets, known to help on imbalanced problems, +0.01–0.02 F1 typical
- [ ] Quick experiment: `Dropout(0.3)` if model is underfitting (val F1 < 0.35)
- [ ] Add MLP layer diagram to slides (boxes with dimensions labeled)

---

## 3. Training Efficiency (5%)

### What we have — DONE
- `FAST_RUN = True/False` flag → 2 epochs (pipeline test) or 30 epochs (real run)
- EarlyStopping saves best checkpoint → stops wasting GPU when val_loss plateaus
- Checkpoint at `task1/outputs/checkpoints/task1_mlp_best.pth`
- Batch size 64 → good GPU utilization on Colab T4, fits in memory
- `pin_memory=True` + `non_blocking=True` → overlaps CPU→GPU data transfer with compute
- `num_workers=2` → parallel data loading

### MUST document in slides (rubric explicitly requires this)
- [ ] **Total training time in minutes** — record from Colab cell output (`elapsed` per epoch)
- [ ] **GPU type** — Colab T4 (15 GB VRAM); MLP uses <<1 GB
- [ ] **Epochs run before early stopping** — expected 10–20 of 30
- [ ] **Budget check:** MLP budget = 1h; actual expected = 5–10 min GPU

---

## 4. Performance Evaluation (10%)

### Metrics we compute — DONE
- **Macro-averaged F1** — primary metric, matches Kaggle leaderboard scoring
- **Accuracy** — tracked per epoch in `evaluate()`
- **Per-class precision / recall / F1** — `classification_report_str()` from sklearn
- **Confusion matrix** (row-normalised) — `plot_confusion_matrix()` seaborn heatmap
- **Training curves** (train_loss, val_loss, val_F1) — `plot_history()`

### Why macro-F1 not accuracy
- Always-predict-Water baseline: ~19% accuracy but macro-F1 ≈ 0.021
- Macro-F1 weights each class equally regardless of sample count → Ground (6.8%) matters as much as Water (18.7%)
- Competition uses macro-F1 → this is the only number that matters

### Baselines (fill in after run)
| Model                           | Macro-F1     |
| ------------------------------- | ------------ |
| Random guess                    | ~0.111 (1/9) |
| Always predict majority (Water) | ~0.021       |
| Our MLP (val)                   | TODO         |
| Our MLP (Kaggle test)           | TODO         |
| Expected range                  | 0.35–0.50    |

### TODO before slides
- [ ] Run full training on Colab (FAST_RUN=False), capture: best val macro-F1, per-class F1, Kaggle score
- [ ] Save `task1_history.png` and `task1_confusion.png` from `task1/outputs/plots/`
- [ ] Interpret confusion matrix: which classes are confused and why
- [ ] Interpret per-class F1: Ground + Rock expected worst → motivates CNN
- [ ] If val F1 < 0.30: try label_smoothing=0.1, re-run

### Nice-to-have (for max score on "in-depth interpretation")
- Precision-Recall curve per class (`sklearn.metrics.precision_recall_curve`)
- Per-class accuracy table alongside per-class F1
- Statement about what macro-F1 of X means practically ("model is correct on X% of types on average, across all classes equally")

---

## 5. Interesting Things to Mention (Presentation "Tell other interesting things" slide)

- **ImageNet normalization on Pokémon:** confirmed close match via `plot_pixel_statistics` → not just a blind default, we actually checked
- **Namespace collision fix:** `src/data/` → `src/datasets/` — Python's implicit namespace packages would have shadowed the `data/` folder on Colab. This is a non-obvious real engineering problem we solved
- **No softmax at output:** `CrossEntropyLoss` = `log_softmax + NLLLoss` internally — adding `nn.Softmax()` before it would compute softmax twice → wrong gradients
- **Stratified split:** `train_test_split(stratify=labels)` preserves 2.76× ratio in both splits — random split could under-represent Ground/Rock in val → misleadingly high val F1
- **`set_seed()` covers all randomness sources:** `random`, `numpy`, `torch`, `torch.cuda`, `cudnn.deterministic=True` — fully reproducible
- **`FAST_RUN` flag:** single boolean in `config.py` switches between 2-epoch pipeline test and 30-epoch real run — no code changes needed to run locally vs Colab

---

## 6. Quick Wins Before Final Submission (Low Effort)

| Experiment                                       | Expected gain                         | Effort               |
| ------------------------------------------------ | ------------------------------------- | -------------------- |
| `label_smoothing=0.1` in CrossEntropyLoss        | +0.01–0.02 macro-F1                   | 1 line in notebook   |
| `Dropout(0.3)` if underfitting                   | Less regularization = more capacity   | 1 line in mlp.py     |
| `EPOCHS=40, PATIENCE=7` if still improving at 30 | More training if loss not flat        | 2 lines in config.py |
| `LR=5e-4` if loss is noisy                       | Smaller steps = more stable           | 1 line in config.py  |
| `batch_size=32`                                  | Noisier gradients can help generalize | 1 line in config.py  |

---

## 7. TODO Checklist Before Presentation

- [ ] Run notebook on Colab (FAST_RUN=False, 30 epochs)
- [ ] Fill in all "Finding" TODO cells in the notebook
- [ ] Record: total training time, GPU, final val macro-F1, Kaggle leaderboard score
- [ ] Save plots to `task1/outputs/plots/` — use in slides
- [ ] Try `label_smoothing=0.1` quick experiment
- [ ] Build slides (structure: Problem → EDA → Architecture → Training → Results → Limitations → Next)
- [ ] Write per-class F1 table in slides
- [ ] Peer review two other groups using the rubric above
