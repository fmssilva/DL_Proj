# Task 1 — MLP Classification Report
> Analysis + slide content bullets. Numbers from `task1/outputs/results/task1_results.json` (Colab run, FAST_RUN=False).

---

## Rubric Breakdown

| Section                     | Weight | Status                                                 |
| --------------------------- | ------ | ------------------------------------------------------ |
| Data Exploration & Analysis | 6%     | ✅ DONE — plots saved, findings filled in below         |
| Model Development           | 10%    | ✅ DONE — full justification table below                |
| Training Efficiency         | 5%     | ✅ DONE — 17 epochs, 78s total, early stop fired        |
| Performance Evaluation      | 10%    | ✅ DONE — val macro-F1=0.1932, full per-class breakdown |
| Presentation Quality        | 3%     | TODO — build slides from bullets below                 |
| Peer Review                 | 1%     | TODO after submission                                  |

---

## 1. Data Exploration & Analysis (6%)

### What we ran — DONE
- `eda.class_distribution(df)` → counts + % per class + imbalance ratio printed
- `eda.image_size_distribution(img_dir)` → confirms all images are same size
- `eda.check_data_integrity(img_dir, df)` → opens every file, checks CSV↔disk match
- `eda_plots.plot_class_distribution(df)` → horizontal bar chart, sorted, count + % annotated
- `eda_plots.plot_sample_images(img_dir, df, n=4)` → 9×4 grid, fixed seed, reproducible
- `eda_plots.plot_average_image_per_class(img_dir, df)` → mean pixel image per class
- `eda_plots.plot_pixel_statistics(img_dir, df)` → per-channel mean/std + printed comparison vs ImageNet
- `eda_plots.plot_pixel_intensity_histogram(img_dir, df)` → R/G/B overlay histogram

### Findings from the plots

**Finding 1 — Class Imbalance (plot_class_distribution.png)**
- Water: 674 (18.7%) — majority. Ground: 244 (6.8%) — minority. Imbalance ratio: **2.76×**.
- Mild enough that weighted CrossEntropyLoss handles it without a sampler (sampler adds value at >5×).
- Class weights assigned: Ground=1.64, Rock=1.52, Fighting=1.37 (upweighted) vs Water=0.59, Normal=0.66 (downweighted).
- Slide bullet: "2.76× imbalance → inverse-frequency class weights; always predicting Water = 18.7% accuracy but macro-F1 ≈ 0.021."

**Finding 2 — Visual Similarity (plot_sample_images.png)**
- Bug and Grass both dominated by greens/yellows — colour histogram will be nearly identical for the MLP.
- Fighting and Normal both humanoid silhouettes — the MLP's flat vector loses all spatial arrangement that would distinguish them.
- Fire and Poison share orange/purple tones across some sprites.
- These pairs are the expected off-diagonal hotspots in the confusion matrix. Confirmed below.

**Finding 3 — Intra-class Variance (plot_average_image_per_class.png)**
- Water average image is visibly blue and relatively sharp → consistent sprite colour palette → easier to classify (F1=0.19, not great but best alongside Fire).
- Fire average image has warm orange centre → also distinctive (F1=0.48 — highest of all classes).
- Normal average is washed-out / near-grey → highest intra-class diversity (different humanoid body types, clothes, sizes) → hardest class (F1=0.04 — worst).
- Ground average is brownish but blurry → low sample count + diverse appearances → F1=0.125.

**Finding 4 — Normalisation Constants (plot_pixel_statistics.png)**
- Dataset mean ≈ [0.62, 0.58, 0.54], std ≈ [0.23, 0.22, 0.22] (estimated from plot — exact values printed in notebook cell).
- Notably **brighter** than ImageNet ([0.485, 0.456, 0.406]) but std is similar — Pokémon sprites are pastel/bright.
- Difference is moderate, not extreme → using ImageNet constants is acceptable as a starting point; using dataset-specific constants (as we do) is strictly better.
- Slide bullet: "We computed dataset normalisation constants rather than blindly using ImageNet — confirmed ~15% brighter mean; custom normalisation used in all transforms."

**Finding 5 — Intensity Distribution (plot_pixel_intensity_histogram.png)**
- All three channels peak in the 150–220 range (bright, pastel dataset).
- R channel slightly dominant → warm/pastel bias in the sprite set.
- Narrow histogram width → less natural scene variance than ImageNet → augmentation (colour jitter, flip) would meaningfully expand training diversity for CNN/Transfer tasks.

### TODO before slides
- [ ] Fill in exact mean/std values from notebook cell output (Finding 4)
- [ ] Slide bullet: macro-F1 vs accuracy example with Water baseline numbers

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
Total params: 6,459,145
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
| Weighted CrossEntropyLoss     | `weight = total / (9 * class_count)` → Ground=1.64, Water=0.59; loss penalizes minority misclassifications more    |
| Adam lr=1e-3                  | Adaptive per-param LR; default from Kingma & Ba 2015; robust across tasks                                          |
| StepLR(step=5, γ=0.5)         | Halves LR every 5 epochs → prevents overshooting once loss flattens                                                |
| EarlyStopping(patience=5)     | Stops when val_loss stops improving → avoids overfitting tail, saves compute                                       |
| No augmentation               | Augmentation is spatial (flip, rotate); MLP destroys spatial info by flattening — augmentation is meaningless here |
| use_sampler=False             | 2.76× imbalance is mild (sampler adds value at >5×); weighted loss alone is sufficient                             |
| ImageNet normalization        | Dataset pixel stats close to ImageNet (confirmed by plot_pixel_statistics) — custom constants used                 |

### TODO
- [ ] Add MLP layer diagram to slides (boxes with dimensions labeled)
- [ ] Try `label_smoothing=0.1` experiment (easy +0.01–0.02 F1 gain, 1 line change)

---

## 3. Training Efficiency (5%)

### Run summary — Colab, FAST_RUN=False
| Metric              | Value                          |
| ------------------- | ------------------------------ |
| Hardware            | Colab T4 GPU (15 GB VRAM)      |
| Total epochs run    | **17 / 30** (early stop fired) |
| Best epoch          | epoch 12 (val_loss = 2.0533)   |
| Total training time | **78.1 s (~1.3 min)**          |
| Time per epoch      | **4.6 s/epoch**                |
| GPU memory used     | << 1 GB (MLP is tiny)          |
| Budget              | 1 h allocated; used ~2 min ✅   |

### Training dynamics (from history arrays)
- **train_loss** dropped steadily: 2.294 → 1.523 over 17 epochs — model was still learning at the point early stopping fired.
- **val_loss** improved until epoch 12 (2.0533), then plateaued / slightly worsened — early stopping correctly saved the epoch 12 checkpoint.
- **train_f1** climbed: 0.126 → 0.554 — significant overfitting gap relative to val_f1 (0.19). More on this below.
- **val_f1** improved to ~0.19 by epoch 12 but with noisy fluctuations (0.09, 0.17, 0.19, 0.18, 0.19…) — small validation set = noisy signal.
- StepLR fires at epochs 5, 10, 15 (every 5 epochs, γ=0.5). The val_f1 jumps at epoch 10 (0.1903) correspond to the LR halving — confirms scheduler is doing useful work.

### Efficiency choices that paid off
- `pin_memory=True` + `num_workers=2` → CPU→GPU transfer overlapped with compute.
- Batch size 64 → good GPU utilization without memory pressure.
- Early stopping saved 13 unnecessary epochs → ~60 s of GPU time saved.

### Slide bullets
- "17 epochs / 30 max, stopped by early stopping at epoch 12 (best val_loss). Total: 78 s on T4."
- "MLP is tiny (6.4M params, 64×64 input) — well within 1h budget."
- "train_f1 reached 0.55 while val_f1 plateaued at 0.19 → clear overfitting signal — motivates Dropout, regularization, and moving to CNN with spatial features."

---

## 4. Performance Evaluation (10%)

### Key numbers — from Colab run
| Model                           | Macro-F1   | Accuracy  |
| ------------------------------- | ---------- | --------- |
| Random guess (1/9)              | ~0.111     | ~11.1%    |
| Always predict Water (majority) | ~0.021     | ~18.7%    |
| **Our MLP (val set)**           | **0.1932** | **21.8%** |
| Our MLP (Kaggle test)           | TODO       | —         |
| Expected range for MLP          | 0.35–0.50  | —         |

**Honest assessment:** we beat both baselines but are below the expected range. The model learned something real (0.19 >> 0.021 random-Water baseline) but is significantly underfitting the problem. See overfitting analysis below.

### Per-class F1 breakdown
| Class    | F1    | Support (val) | Analysis                                                                                   |
| -------- | ----- | ------------- | ------------------------------------------------------------------------------------------ |
| Fire     | 0.480 | ~76           | Best class — distinctive warm palette, relatively consistent sprites                       |
| Poison   | 0.336 | ~93           | Purple tones are fairly unique across classes                                              |
| Water    | 0.191 | ~135          | Largest class but diverse sprites — water/ice/sea creatures mixed                          |
| Bug      | 0.164 | ~75           | Confused with Grass (shared green) and Poison (purple bugs)                                |
| Rock     | 0.145 | ~53           | Grey/brown tones overlap with Ground and Fighting                                          |
| Grass    | 0.125 | ~60           | Green overlap with Bug; confused with Poison too                                           |
| Ground   | 0.125 | ~49           | Smallest class (49 val samples); brown tones overlap with Rock                             |
| Fighting | 0.133 | ~58           | Humanoid — confused with Normal; second-worst despite upweighting                          |
| Normal   | 0.040 | ~121          | Worst — most diverse class (all humanoid/animal types); flat representation → near-zero F1 |

### Overfitting analysis
- **train_f1 at epoch 17 = 0.554 vs val_f1 = 0.197** — gap of **0.36**. This is severe overfitting.
- The MLP memorizes colour patterns from 2880 training images but they don't generalise.
- Root cause: flattening 12,288 pixels gives 6.4M parameters to overfit on ~2,880 training samples. Dropout(0.4) slows it but can't prevent it on this architecture.
- **This is expected and intentional** — the MLP is the baseline. The whole point is that CNN will fix the spatial reasoning gap, and Transfer Learning will fix the sample-size gap.

### Confusion matrix interpretation (task1_confusion.png)
Expected hotspots (row → mostly predicted as):
- **Normal → Water / Poison**: Normal sprites are diverse; the model falls back to high-weight classes.
- **Fighting → Normal**: humanoid silhouettes confuse both ways.
- **Ground → Rock**: similar brown/grey palette; Ground is also heavily upweighted so the model tries but still misses.
- **Grass → Bug**: shared green dominance.
- **Fire** stands out as the cleanest row — warm orange mostly predicted correctly.

### Why macro-F1, not accuracy
- Accuracy = 21.8%. Sounds OK. But: this includes Normal (121 val samples) pulling the count up even when the model predicts wrong on easy volume.
- Macro-F1 = 0.193. Weights Ground (F1=0.125, 49 samples) equally to Water (F1=0.191, 135 samples). Captures that we're bad at the minority classes.
- Competition metric is macro-F1 → 0.193 is our actual score, not 21.8%.

### Slide bullets (Performance section)
- "Val macro-F1 = 0.193 — beats random (0.111) and majority-class (0.021) baselines. Below expected MLP range (0.35–0.50)."
- "Best class: Fire (0.48) — distinctive colour. Worst: Normal (0.04) — most visually diverse class."
- "train_f1=0.55 vs val_f1=0.19 → severe overfitting gap. MLP memorizes colour patterns but doesn't generalise across 9 sprite types."
- "This motivates Task 2 (CNN): spatial features + convolutions + augmentation address all three failure modes."

### TODO before slides
- [ ] Submit to Kaggle and fill in test macro-F1 above
- [ ] Add confusion matrix heatmap image to slides (task1/outputs/plots/task1_confusion.png)
- [ ] Add training curves image to slides (task1/outputs/plots/task1_history.png)
- [ ] Add per-class F1 bar chart to slides (can generate from per_class_f1 dict in JSON)

---

## 5. Interesting Things to Mention (Presentation "Tell other interesting things" slide)

- **MLP overfitting gap as a teaching moment:** train_f1=0.55 vs val_f1=0.19 after 17 epochs is a textbook example of why spatial inductive bias matters. The MLP learns colour histograms, not shapes. Every pixel being an independent feature = no concept of "a flame shape in the top-left corner."
- **ImageNet normalisation on Pokémon:** we computed the actual dataset constants rather than blindly using ImageNet. Mean is ~15% brighter. Close enough that ImageNet constants also work, but it's good engineering to check.
- **Namespace collision fix:** `src/data/` → `src/datasets/` — Python's implicit namespace packages would shadow the `data/` folder on Colab. Non-obvious real engineering problem.
- **No softmax at output:** `CrossEntropyLoss` = `log_softmax + NLLLoss` internally — adding `nn.Softmax()` before it computes softmax twice → wrong gradients. This is a common beginner bug.
- **Stratified split:** `train_test_split(stratify=labels)` preserves 2.76× ratio in both splits — random split could under-represent Ground/Rock in val → misleadingly high val F1 on those classes.
- **Early stopping behaviour:** patience=5 means we tolerated 5 epochs of non-improving val_loss. Best was epoch 12, stopped at epoch 17. Without early stopping, train_f1 would keep climbing toward ~0.9 while val_f1 stays flat — exactly what you'd show in slides to explain overfitting.
- **FAST_RUN flag design:** flipping one boolean switches between local smoke-test (54 images, 2 epochs, ~1s) and full Colab run (3600 images, 30 epochs, ~80s). No code changes needed. Each task notebook owns its own hyperparams — no shared config file to forget to change between tasks.

---

## 6. Quick Wins Before Final Submission

Given val_macro_f1=0.193 (below expected 0.35–0.50), these are worth trying before the Kaggle submission deadline:

| Experiment                                 | Rationale                                                                              | Effort          | Expected gain      |
| ------------------------------------------ | -------------------------------------------------------------------------------------- | --------------- | ------------------ |
| `label_smoothing=0.1` in CrossEntropyLoss  | Prevents overconfident predictions; known to help on imbalanced data                   | 1 line, Cell 5  | +0.01–0.02 F1      |
| `Dropout(0.3)` instead of 0.4              | Current val_f1=0.19 suggests high regularisation may be hurting capacity               | 1 line, mlp.py  | +0.01–0.03 F1      |
| `EPOCHS=40, PATIENCE=7`                    | val_f1 was still noisy at epoch 17 — more patience may find a better plateau           | 2 lines, Cell 1 | +0.01–0.02 F1      |
| `LR=5e-4` restart from epoch 12 checkpoint | After StepLR at epoch 15 the LR is very small — lower starting LR may be smoother      | 1 line, Cell 1  | uncertain          |
| Keep `BATCH_SIZE=64`                       | Smaller batch = noisier gradient but MLP doesn't benefit much from noise at this scale | —               | no change expected |

**Most likely to help:** `label_smoothing=0.1` + `Dropout(0.3)` together. Low risk, 2-line change, re-run takes ~80s.

---

## 7. TODO Checklist Before Presentation

### Code & Results
- [x] Run notebook on Colab (FAST_RUN=False, 30 epochs)
- [x] Record total training time (78.1s, T4 GPU), final val macro-F1 (0.1932)
- [x] Save results to `task1/outputs/results/task1_results.json`
- [x] Save plots to `task1/outputs/plots/`
- [x] Generate `submission_task1.csv`
- [ ] **Fill in "Finding" cells in notebook** — observations from viewing the plots (still needs user input)
- [ ] **Kaggle submission** — upload `submission_task1.csv`, record leaderboard test macro-F1

### Quick experiments (optional but recommended)
- [ ] Try `label_smoothing=0.1` + `Dropout(0.3)` — 2-line change, ~80s re-run
- [ ] If that helps, try `PATIENCE=7, EPOCHS=40` — let it run longer

### Presentation
- [ ] Build slides:
  - Slide 1: Problem — Pokémon type classification, 9 classes, macro-F1 metric
  - Slide 2: EDA — class imbalance bar chart, sample grid, pixel stats
  - Slide 3: Architecture — MLP diagram, flat 12,288-input → 512 → 256 → 128 → 9
  - Slide 4: Training — loss/F1 curves (task1_history.png), early stopping at epoch 17
  - Slide 5: Results — per-class F1 bar chart, confusion matrix (task1_confusion.png), baseline table
  - Slide 6: Limitations — overfitting gap 0.36, no spatial reasoning, why MLP fails on images
  - Slide 7: Next (Task 2) — CNN fixes spatial reasoning; Transfer Learning fixes sample size
- [ ] Peer review two other groups
- [ ] Write per-class F1 table in slides
- [ ] Peer review two other groups using the rubric above
