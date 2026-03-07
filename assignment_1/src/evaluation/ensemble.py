# Soft ensemble: average softmax outputs of N pre-trained models at inference time.
# No retraining. Pass a list of (model_instance, checkpoint_path) pairs — that's it.
# Works for any architecture mix across all 3 tasks.

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

from ..config import CLASSES


def soft_ensemble(
    checkpoint_configs: list,
    val_loader: DataLoader,
    device: torch.device,
    classes: list = CLASSES,
    weights: Optional[list] = None,
    inference_mode: bool = False,
) -> dict:
    """
    Load N checkpoints, average their softmax outputs, return metrics.

    checkpoint_configs: list of (nn.Module, str|Path)
        Each tuple is (model_instance_on_cpu, checkpoint_path).
        The model architecture must match the checkpoint — caller's responsibility.

    weights: optional list of floats, one per model, for weighted average.
        None -> uniform average (equal weight per model).
        e.g. [0.6, 0.4] to weight the first model more heavily.

    inference_mode: set True when val_loader is a test loader (no integer labels).
        Skips label collection, F1/acc computation, and per-model solo passes.
        Returns val_macro_f1=None, val_acc=None, per_model_f1={} in this case.

    Returns dict:
        {
          "val_macro_f1": float | None,
          "val_acc":      float | None,
          "per_model_f1": {checkpoint_stem: float},  # empty in inference_mode
          "preds":        list[int],
          "labels":       list[int],                 # empty in inference_mode
        }
    """
    if not checkpoint_configs:
        raise ValueError("checkpoint_configs must contain at least one (model, path) pair")

    if weights is not None:
        if len(weights) != len(checkpoint_configs):
            raise ValueError("weights length must match checkpoint_configs length")
        total = sum(weights)
        weights = [w / total for w in weights]  # normalise to sum to 1
    else:
        w = 1.0 / len(checkpoint_configs)
        weights = [w] * len(checkpoint_configs)

    # load all models once — avoid re-loading on every batch
    loaded_models = []
    per_model_f1  = {}
    for model, ckpt_path in checkpoint_configs:
        ckpt_path = Path(ckpt_path)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.to(device).eval()
        loaded_models.append(model)
        # note: individual F1 computed after ensembling loop via separate pass
        per_model_f1[ckpt_path.stem] = None

    # single pass — accumulate weighted softmax sums
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            # start with zeros, then add each model's weighted contribution
            avg = torch.zeros(imgs.size(0), len(classes), device=device)
            for model, w in zip(loaded_models, weights):
                avg += w * F.softmax(model(imgs), dim=1)
            all_preds.extend(avg.argmax(dim=1).cpu().tolist())
            # labels is a tensor on val sets, but a tuple of strings on test set
            if not inference_mode:
                all_labels.extend(labels.tolist())

    if inference_mode:
        # test loader — no ground-truth labels, skip metrics and per-model pass
        return {
            "val_macro_f1": None,
            "val_acc":      None,
            "per_model_f1": {},
            "preds":        all_preds,
            "labels":       [],
        }

    ens_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    ens_acc = accuracy_score(all_labels, all_preds)

    # compute individual F1 for each model (cheap second pass — one loader each)
    for model, (_, ckpt_path) in zip(loaded_models, checkpoint_configs):
        solo_preds, solo_labels = [], []
        # reuse the same val_loader — it resets automatically each iteration
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                solo_preds.extend(model(imgs).argmax(dim=1).cpu().tolist())
                solo_labels.extend(labels.tolist())
        stem = Path(ckpt_path).stem
        per_model_f1[stem] = round(
            f1_score(solo_labels, solo_preds, average="macro", zero_division=0), 4
        )

    return {
        "val_macro_f1": round(ens_f1, 4),
        "val_acc":      round(ens_acc, 4),
        "per_model_f1": per_model_f1,
        "preds":        all_preds,
        "labels":       all_labels,
    }


def print_ensemble_report(result: dict, ensemble_label: str = "Ensemble") -> None:
    """Pretty-print the result dict returned by soft_ensemble."""
    print(f"\n=== {ensemble_label} ===")
    for stem, f1 in result["per_model_f1"].items():
        print(f"  {stem:<30} solo val_macro_f1 = {f1:.4f}")
    print(f"  {'--- ensemble ---':<30} val_macro_f1 = {result['val_macro_f1']:.4f}  "
          f"val_acc = {result['val_acc']:.4f}")
    # delta vs best individual
    best_solo = max(result["per_model_f1"].values())
    delta     = result["val_macro_f1"] - best_solo
    print(f"  Delta vs best solo  : {delta:+.4f}")


# ── local test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from src.config import CLASSES

    device    = torch.device("cpu")
    n_classes = len(CLASSES)
    tmp_dir   = Path(tempfile.mkdtemp())

    # two tiny models — same architecture, random weights
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3 * 64 * 64, n_classes)
        def forward(self, x):
            return self.fc(x.view(x.size(0), -1))

    m1, m2 = TinyModel(), TinyModel()
    ckpt1   = tmp_dir / "model_1.pth"
    ckpt2   = tmp_dir / "model_2.pth"
    torch.save(m1.state_dict(), ckpt1)
    torch.save(m2.state_dict(), ckpt2)

    # dummy val loader
    imgs   = torch.randn(16, 3, 64, 64)
    labels = torch.randint(0, n_classes, (16,))
    loader = DataLoader(TensorDataset(imgs, labels), batch_size=8)

    # --- test uniform ensemble ---
    result = soft_ensemble([(TinyModel(), ckpt1), (TinyModel(), ckpt2)], loader, device)
    assert 0.0 <= result["val_macro_f1"] <= 1.0
    assert 0.0 <= result["val_acc"] <= 1.0
    assert len(result["preds"]) == 16
    assert set(result["per_model_f1"].keys()) == {"model_1", "model_2"}
    print("[PASS] uniform ensemble returned correct keys and ranges")

    # --- test weighted ensemble ---
    result_w = soft_ensemble(
        [(TinyModel(), ckpt1), (TinyModel(), ckpt2)], loader, device, weights=[0.7, 0.3]
    )
    assert 0.0 <= result_w["val_macro_f1"] <= 1.0
    print("[PASS] weighted ensemble ran without error")

    print_ensemble_report(result, "Test Ensemble")

    # --- test error on mismatched weights ---
    try:
        soft_ensemble([(TinyModel(), ckpt1)], loader, device, weights=[0.5, 0.5])
        print("[FAIL] should have raised ValueError for mismatched weights")
    except ValueError as e:
        print(f"[PASS] ValueError raised correctly: {e}")

    print("\nAll ensemble.py smoke tests passed.")
