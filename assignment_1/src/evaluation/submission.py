# Submission CSV generation and validation for Kaggle.
# Expects the test DataLoader to run in inference mode (csv_path=None).

from pathlib import Path

import pandas as pd
import torch

from ..config import CLASSES


def generate_submission(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    label_map: list,
    out_path: Path,
    device: torch.device,
) -> None:
    """
    Run inference on the test set and write a Kaggle submission CSV.
    label_map is just CLASSES — label_map[i] gives the string name for integer prediction i.
    test_loader must yield (tensor, uuid_string) batches (inference mode).
    """
    model.eval()
    ids, preds = [], []

    with torch.no_grad():
        for images, uuids in test_loader:
            images  = images.to(device)
            logits  = model(images)
            pred_idx = logits.argmax(dim=1).cpu().tolist()
            preds.extend([label_map[i] for i in pred_idx])
            # uuids comes as a tuple of strings from DataLoader collation
            ids.extend(list(uuids))

    df = pd.DataFrame({"Id": ids, "label": preds})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}  ({len(df)} rows)")


def generate_submission_from_preds(
    test_loader: torch.utils.data.DataLoader,
    preds: list,
    label_map: list,
    out_path: Path,
) -> None:
    """
    Write a Kaggle submission CSV from already-computed integer predictions.
    Use this when predictions come from an ensemble or were collected manually —
    no model forward pass needed here, just UUID collection + CSV write.

    test_loader must yield (tensor, uuid_string) batches (inference mode).
    preds: list of integer class indices, one per test image, in loader order.
    label_map: list of class name strings — label_map[i] gives the name for index i.
    """
    ids = []
    for _, uuids in test_loader:
        ids.extend(list(uuids))

    if len(ids) != len(preds):
        raise ValueError(f"Mismatch: {len(ids)} test images but {len(preds)} predictions")

    df = pd.DataFrame({"Id": ids, "label": [label_map[i] for i in preds]})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}  ({len(df)} rows)")


def validate_submission(path: Path, expected_rows: int = 900) -> None:
    """
    Sanity-check the submission file before uploading.
    Raises ValueError with a clear message if anything looks wrong.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Submission file not found: {path}")

    df = pd.read_csv(path)

    # header check
    if list(df.columns) != ["Id", "label"]:
        raise ValueError(f"Expected columns ['Id', 'label'], got {list(df.columns)}")

    # row count
    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")

    # all predicted class names must be valid
    invalid = set(df["label"]) - set(CLASSES)
    if invalid:
        raise ValueError(f"Unknown class names in submission: {invalid}")

    # no missing values
    if df.isnull().any().any():
        raise ValueError("Submission contains NaN values")

    print(f"Submission valid: {len(df)} rows, all class names correct.")


# ── local tests ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    from torch.utils.data import TensorDataset, DataLoader as _DL
    sys.path.insert(0, str(Path(__file__).parents[2]))

    tmp = Path(tempfile.mkdtemp())

    # --- test validate_submission: valid case ---
    dummy_rows = [{"Id": f"uuid-{i:04d}", "label": CLASSES[i % len(CLASSES)]} for i in range(900)]
    df_dummy = pd.DataFrame(dummy_rows)
    tmp_path = tmp / "sub_valid.csv"
    df_dummy.to_csv(tmp_path, index=False)
    validate_submission(tmp_path, expected_rows=900)
    print("[PASS] validate_submission: valid submission passes")

    # --- test validate_submission: wrong row count ---
    tmp_bad = tmp / "sub_bad.csv"
    df_dummy.iloc[:50].to_csv(tmp_bad, index=False)
    try:
        validate_submission(tmp_bad, expected_rows=900)
        print("[FAIL] should have raised for wrong row count")
    except ValueError as e:
        print(f"[PASS] validate_submission: correctly raised for wrong row count")

    # --- test generate_submission_from_preds ---
    # simulate test_loader yielding (tensor, uuid_string) batches
    dummy_imgs   = torch.randn(20, 3, 64, 64)
    dummy_uuids  = [f"uuid-{i:04d}" for i in range(20)]
    # DataLoader with string targets: we need a custom collate — simulate manually
    test_batches = [(dummy_imgs[:10], dummy_uuids[:10]), (dummy_imgs[10:], dummy_uuids[10:])]

    class _FakeLoader:
        def __iter__(self):
            return iter(test_batches)

    preds_int = [i % len(CLASSES) for i in range(20)]
    sub_path  = tmp / "sub_from_preds.csv"
    generate_submission_from_preds(_FakeLoader(), preds_int, CLASSES, sub_path)
    assert sub_path.exists()
    df_out = pd.read_csv(sub_path)
    assert list(df_out.columns) == ["Id", "label"]
    assert len(df_out) == 20
    assert df_out.iloc[0]["label"] == CLASSES[0]
    print("[PASS] generate_submission_from_preds wrote correct CSV")

    # --- mismatch raises ---
    try:
        generate_submission_from_preds(_FakeLoader(), preds_int[:5], CLASSES, sub_path)
        print("[FAIL] should have raised for mismatched preds length")
    except ValueError:
        print("[PASS] generate_submission_from_preds raises on mismatched length")

    print("\nAll submission.py smoke tests passed.")
