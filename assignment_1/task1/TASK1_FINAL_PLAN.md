# TASK1_FINAL_PLAN.md — How We Complete Task 1

> **Goal:** Reach the best possible val macro-F1 with an MLP on flat pixels.
> **Metric:** Macro-averaged F1. Random baseline = 0.111. Our first run = **0.1932**.
> **Budget:** ~1 h GPU total. One Colab run ≈ 80s. 5 experiments ≈ 8 min total.
> **Status:** Implementation complete. All src/ changes done. Notebook rebuilt. Ready for Colab run.

---


## What We Know Going In

From the first Colab run (task1_results.json):

- val macro-F1 = **0.193** (barely above random 0.111)
- train macro-F1 = **0.554** — gap of **0.36** → severe overfitting
- best epoch = 12 / 30, early stopping at epoch 17 (patience=5)
- Fire = best class (F1=0.48, distinctive colour). Normal = worst (F1=0.04, most diverse).
- MLP with flat pixels **cannot** learn spatial structure → this is expected and intentional

The MLP is a baseline. Our job is to push it as far as it can go, record what works, then hand off to CNN (Task 2) with a clear story.

---

## Part 0 — EDA Improvements (Phase 1 of __todo.md)

» Q/T: were these 3 points already added to the  EDA section of the notebook? 

1. **Normalization from training set** — explicitly compute and print the dataset mean/std from the train split (not the full set). Confirm they're close to ImageNet values. One code cell after the data loaders are built. This is a data correctness story for the report.

2. **PCA / t-SNE cluster plot** — flatten each image to 12,288 features, run PCA to 50 dims then t-SNE to 2D, colour by class. Shows whether any classes are linearly separable in pixel space. Expected: Fire and Water will have some cluster structure; Normal will be scattered everywhere. This directly motivates why MLP struggles (no clean clusters in pixel space).

3. **Label correctness spot-check** — already done implicitly (all 3600 load correctly), but worth printing 2 examples per class in the sample grid with their label above. Already exists in `plot_sample_images`.

---


## Part 2 — Iterative Model Improvements (Phases 3–6 of __todo.md)

| #   | What                                             | Why                                                       | Where to change      | Expected effect                                              |
| --- | ------------------------------------------------ | --------------------------------------------------------- | -------------------- | ------------------------------------------------------------ |
| A   | Vanilla baseline                                 | reference point                                           | model instantiation  | val_f1 ≈ 0.13 (worse, confirms our design is already better) |
| B   | Dropout 0.4→0.3                                  | slightly less regularisation may help capacity            | `mlp.py` or inline   | ±0.01–0.02                                                   |
| C   | `label_smoothing=0.1` in CrossEntropyLoss        | softens overconfident predictions, helps minority classes | criterion definition | +0.01–0.03                                                   |
| D   | B+C together                                     | combined effect                                           | both above           | +0.02–0.05 (best expected combo)                             |
| E   | `weight_decay=1e-4` in Adam                      | L2 reg, reduces overfit                                   | optimizer definition | +0.01–0.02                                                   |
| F   | `use_sampler=True` (WeightedRandomSampler)       | oversamples minority classes during training              | loader call          | may help Ground/Rock F1                                      |
| G   | `augment=True` (horizontal flip + colour jitter) | more training diversity                                   | loader call          | small effect for MLP (spatial meaning lost)                  |

**Rule: one change at a time.** Track every experiment result.

### What NOT to try
- More layers or wider layers → already overfitting, more params makes it worse
- `EPOCHS=50` → val_f1 plateaued at epoch 8, more epochs won't help
» Q/T: confirm we have some notes about this in the notebook 

» Q/T: we have the vanila and the baseline and then we try other models... shouldn't we also try different architectures of MLP? example more deep with narrow layers to not overfit or with some bottlenecks or skip connections?? if so, when to do this? before or after all the other regularisation experiments?
» Q/T: confirm we have some notes about this in the notebook 


## On "Grid Search vs Manual"
- Grid search over 5×5 params = 25 runs × 80s = ~33 min. Not worth it.
- We know the problem is overfitting. The relevant axis is **regularisation strength** (dropout, weight_decay, label_smoothing), not architecture search.
- 4–5 manual targeted experiments, each motivated by the previous result, will learn more than a blind grid search.
- If we had more time: `torch.nn.utils.clip_grad_norm_` as Experiment F (gradient clipping), and `ReduceLROnPlateau` instead of StepLR as Experiment G.
» Q/T: confirm we have some notes about this in the notebook 




## Part 3 — Data Augmentation (Phase 7 of __todo.md)

**For MLP:** augmentation is largely meaningless. RandomHorizontalFlip changes which pixel is at position (0,0) but the MLP sees it as a completely different flat vector. The model can't learn that flipping = same object.

**Verdict:** Try `augment=True` as Experiment E — include it for completeness and to confirm empirically that it doesn't help MLP. This makes the Task 1 → Task 2 handoff story cleaner: "augmentation doesn't help MLP but will help CNN."

» Q/T: even if we won't have good results with this, lets implement and test with some data augmentation... this way i prepare the pipeline structure for the next tasks and confirm everything works ok, and also i actually demonstrate in the notebook that it is not worth it. 


» Q/T: so we can have: 
```
first cells... 
── SECTION: EDA (always on df_full — full 3600 images) ──────
── SECTION: Part 2 — Shared helpers (run once) ──────────────
── SECTION: Part 3 — Comparison & Final Model ───────────────
Cell 27 — "Part 3" section markdown
Cell 28 — Results table + bar chart (experiment comparison plot saved)
Cell 29 — Load best experiment + full classification report
Cell 30 — Training curves + confusion matrix for best experiment

── SECTION: Part 4 — Best Model with data augmentation ───────────────
we take the best model and run with data augmentation to confirm it doesn't help MLP. This is a sanity check and also prepares the pipeline for Task 2.
and we can have some plot or something to confirm results... 

── SECTION: Save + Submit ────────────────────────────────────
Cell 31 — Save task1_results.json (all experiments + per_class_f1 + config)
Cell 32 — Generate + validate submission_task1.csv
Cell 33 — Download outputs zip (Colab only)
Cell 34 — Summary table (fill in after run: best exp, F1, GPU, time, confusion pair)
```

---

**Test set:** never used during training or model selection. Only touched in the submission cell.

---