# Task 1 entry-point: EDA + train MLP + evaluate + generate submission.
# Pure orchestration — no model/transform/metric logic lives here.

import pandas as pd
import torch
import torch.nn as nn

from src.config import (
    BATCH_SIZE, CLASSES, DATA_DIR, EPOCHS, IMG_SIZE_SMALL,
    LR, NUM_WORKERS, OUT_DIR, PATIENCE, SEED,
    create_output_dirs, set_seed,
)
from src.datasets.dataset import (
    PokemonDataset,
    compute_class_weights,
    get_base_transforms,
    get_train_val_loaders,
)
import src.datasets.eda as eda
import src.datasets.eda_plots as eda_plots
from src.evaluation.metrics import classification_report_str
from src.evaluation.plots import plot_confusion_matrix, plot_history
from src.evaluation.submission import generate_submission, validate_submission
from src.models.mlp import MLP
from src.training.early_stopping import EarlyStopping
from src.training.train import evaluate, run_epoch


CSV_PATH  = DATA_DIR / "train_labels.csv"
TRAIN_DIR = DATA_DIR / "Train"
TEST_DIR  = DATA_DIR / "Test"
CKPT_PATH = OUT_DIR / "checkpoints" / "task1_mlp_best.pth"
SUB_PATH  = OUT_DIR / "results"     / "submission_task1.csv"


def main():
    set_seed(SEED)
    create_output_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── EDA ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    eda.class_distribution(df)
    eda.image_size_distribution(TRAIN_DIR)
    eda.check_data_integrity(TRAIN_DIR, df)

    eda_plots.plot_class_distribution(df)
    eda_plots.plot_sample_images(TRAIN_DIR, df, n_per_class=4)
    eda_plots.plot_average_image_per_class(TRAIN_DIR, df)
    eda_plots.plot_pixel_statistics(TRAIN_DIR, df)
    eda_plots.plot_pixel_intensity_histogram(TRAIN_DIR, df, n_samples=200)

    # ── data loaders ─────────────────────────────────────────────────────────
    # augment=False + use_sampler=False for MLP:
    # weighted loss alone is enough for 2.76x imbalance; sampler adds complexity with no gain here
    train_loader, val_loader = get_train_val_loaders(
        CSV_PATH, TRAIN_DIR, IMG_SIZE_SMALL, BATCH_SIZE,
        augment=False, use_sampler=False, num_workers=NUM_WORKERS,
    )
    test_ds     = PokemonDataset(TEST_DIR, get_base_transforms(IMG_SIZE_SMALL), csv_path=None)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # ── model, loss, optimiser, scheduler ────────────────────────────────────
    # compute class weights from the full training CSV — not just the loader batch
    df_train = df.copy()  # already loaded above for EDA
    label_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    all_train_labels = [label_to_idx[lbl] for lbl in df_train["label"]]
    class_weights = compute_class_weights(all_train_labels).to(device)

    model     = MLP().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    stopper   = EarlyStopping(patience=PATIENCE, checkpoint_path=str(CKPT_PATH))

    # ── training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss, val_metrics = run_epoch(
            epoch, EPOCHS, model, train_loader, val_loader,
            criterion, optimizer, device,
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        # train F1 not computed per epoch (too slow); store NaN as placeholder for plot
        history["train_f1"].append(float("nan"))
        history["val_f1"].append(val_metrics["macro_f1"])

        stopper(val_metrics["loss"], model)
        if stopper.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    # ── evaluation ───────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    val_metrics = evaluate(model, val_loader, criterion, device)
    print(f"\nBest checkpoint val_loss={val_metrics['loss']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}")

    # collect all val predictions for the report + confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    print("\nPer-class report:")
    print(classification_report_str(all_labels, all_preds, CLASSES))

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_history(history, OUT_DIR / "plots" / "task1_history.png")
    plot_confusion_matrix(all_labels, all_preds, CLASSES, OUT_DIR / "plots" / "task1_confusion.png")

    # ── submission ────────────────────────────────────────────────────────────
    generate_submission(model, test_loader, CLASSES, SUB_PATH, device)
    validate_submission(SUB_PATH, expected_rows=900)


if __name__ == "__main__":
    main()
