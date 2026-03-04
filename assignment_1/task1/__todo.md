## Try Unsupervised Learning

Try some unsupervised learning like PCA or something to see if we get good clusters??

---

# 🎮 Pokemon Classifier — Training Checklist

---

## PHASE 1 — Data Sanity Checks (Before Any Training)

- **No data leakage** — train/val/test splits completely separate, zero overlap
- **Stratified splits** — each split has proportional class representation
- **Class distribution** — count images per class, flag imbalanced classes
- **Image quality** — spot corrupted, truncated, unreadable files
- **Consistent format** — all RGB, same size or resizable
- **Label correctness** — manually spot-check a sample
- **Normalization from training set only** — compute mean/std on train; apply same to val/test

---

## PHASE 2 — Baseline Model

- **Architecture:** Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(num_classes, Softmax)
- **Optimizer:** Adam, lr=0.001
- **Loss:** categorical cross-entropy
- **Metric:** accuracy
- **Epochs:** 50 with early stopping
- **Batch size:** 32
- **Pixel normalization:** divide by 255
- **Confirm pipeline works** — loss decreases in first few epochs
- **Save baseline val accuracy** as reference point

---

## PHASE 3 — Monitor Every Experiment

- Plot training vs validation loss each run
- Plot training vs validation accuracy each run
- Check first 5 epochs — if loss not moving, stop and debug
- Loss = NaN → exploding gradients or bad LR, stop immediately
- Flat loss → LR too low or model too simple
- Oscillating loss → LR too high
- Early stopping: stop if val loss doesn't improve for 10 epochs, revert to best weights
- Log every experiment: model version, hyperparams, val accuracy, notes

---

## PHASE 4 — Iterative Improvements (ONE change at a time!)

### Architecture

- Add Conv layers (better for images than just flatten+dense)
- Increase neurons per layer (try 256, 512)
- Use ReLU everywhere in hidden layers

### Regularization

- Dropout after dense layers — try p=0.3 to 0.5
- Weight decay / L2 regularization
- Batch normalization after conv/dense layers, before activation

### Optimizer & LR

- Try different LRs: 0.01, 0.001, 0.0001
- Learning rate scheduler — ReduceLROnPlateau
- Gradient clipping — cap norm at 1.0 to prevent exploding gradients
- Try SGD with momentum=0.9 if Adam plateaus

### Batch size

- Try 16 (more noise, can help generalization)
- Try 128 (faster but may generalize less)

---

## PHASE 5 — Diagnosing Problems

| Symptom                   | Problem                       | Fix                                             |
| ------------------------- | ----------------------------- | ----------------------------------------------- |
| Both losses high          | Underfitting                  | Bigger model, fewer regularization, more epochs |
| Train low, Val high       | Overfitting                   | Dropout, weight decay, augmentation             |
| Loss not moving           | LR too low or broken pipeline | Increase LR, debug data loading                 |
| Loss oscillating          | LR too high                   | Reduce LR                                       |
| Loss = NaN                | Exploding gradients           | Gradient clipping, lower LR                     |
| Val stuck, train improves | Imbalance or leakage          | Check splits, use weighted loss                 |

---

## PHASE 6 — Data Augmentation (After Solid Baseline)

- Horizontal flip
- Random rotation ±15–30°
- Random crop / zoom
- Color jitter (brightness, contrast, saturation)
- Apply ONLY to training set, never val/test
- Visually inspect augmented samples — confirm they look realistic
- Measure val accuracy before and after

---

## PHASE 7 — Class Imbalance

- Weighted loss — higher penalty for underrepresented classes
- Oversampling minority classes
- Use F1 score + confusion matrix, not just accuracy
- Inspect which classes are most confused with each other

---

## PHASE 8 — Final Evaluation

- Evaluate on test set only once at the very end
- Report accuracy, F1, confusion matrix
- Inspect worst performing classes — often reveals data issues
- Save best model checkpoint

---

## Quick Reference Hyperparameters

| Param                   | Start Value               |
| ----------------------- | ------------------------- |
| Optimizer               | Adam                      |
| Learning rate           | 0.001                     |
| Batch size              | 32                        |
| Epochs                  | 50 + early stopping       |
| Early stopping patience | 10 epochs                 |
| Dropout                 | 0.3–0.5                   |
| Pixel normalization     | ÷ 255                     |
| Loss                    | Categorical cross-entropy |
| Output activation       | Softmax                   |
| Hidden activation       | ReLU                      |

---

**Golden rule:** change ONE thing at a time. Log everything. Trust your validation loss, not your training loss.