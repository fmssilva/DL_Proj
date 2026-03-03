# Local tests for train.py — run on CPU before touching Colab.
# Run with: python src/training/train_test.py  (from assignment_1/ root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.train import evaluate, train_one_epoch


def _make_tiny_loader(n_batches: int = 4, batch_size: int = 8, n_classes: int = 9):
    """Build a tiny random DataLoader that fits on CPU in milliseconds."""
    x = torch.randn(n_batches * batch_size, 16)   # 16-dim flat input
    y = torch.randint(0, n_classes, (n_batches * batch_size,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def test_loss_decreases():
    """Loss should drop after 10 training steps on a tiny batch (basic sanity)."""
    device   = torch.device("cpu")
    loader   = _make_tiny_loader()
    model    = nn.Linear(16, 9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # record loss before any training
    initial_metrics = evaluate(model, loader, criterion, device)
    initial_loss    = initial_metrics["loss"]

    # run 10 epochs to get a reliable signal (2 steps is too noisy with random weights)
    for _ in range(10):
        train_one_epoch(model, loader, criterion, optimizer, device)

    final_metrics = evaluate(model, loader, criterion, device)
    final_loss    = final_metrics["loss"]

    assert final_loss < initial_loss, \
        f"Loss did not decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

    print(f"[PASS] loss decreased: {initial_loss:.4f} -> {final_loss:.4f}")


def test_evaluate_output_shape():
    """evaluate() must return a dict with exactly the right keys and sane value ranges."""
    device    = torch.device("cpu")
    loader    = _make_tiny_loader()
    model     = nn.Linear(16, 9).to(device)
    criterion = nn.CrossEntropyLoss()

    metrics = evaluate(model, loader, criterion, device)

    required_keys = {"loss", "acc", "macro_f1"}
    assert set(metrics.keys()) == required_keys, \
        f"Expected keys {required_keys}, got {set(metrics.keys())}"

    # loss can be > 1 for untrained model, but acc and f1 must be in [0, 1]
    assert 0.0 <= metrics["acc"]      <= 1.0, f"acc out of range: {metrics['acc']}"
    assert 0.0 <= metrics["macro_f1"] <= 1.0, f"macro_f1 out of range: {metrics['macro_f1']}"
    assert metrics["loss"] > 0.0, f"loss should be positive, got {metrics['loss']}"

    print(f"[PASS] evaluate() keys and value ranges correct: {metrics}")


def test_train_fp32_and_no_scaler():
    """Confirm scaler=None path runs without error and returns a float."""
    device    = torch.device("cpu")
    loader    = _make_tiny_loader()
    model     = nn.Linear(16, 9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    loss = train_one_epoch(model, loader, criterion, optimizer, device, scaler=None)
    assert isinstance(loss, float), f"Expected float, got {type(loss)}"
    assert loss > 0.0, f"Loss should be positive, got {loss}"

    print(f"[PASS] train_one_epoch (scaler=None) returned loss={loss:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("train_test.py — running all tests")
    print("=" * 60)
    test_evaluate_output_shape()
    test_train_fp32_and_no_scaler()
    test_loss_decreases()
    print("=" * 60)
    print("All training tests passed.")
    print("=" * 60)
