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


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    # build a dummy submission with 900 rows and check it passes validation
    dummy_rows = [{"Id": f"uuid-{i:04d}", "label": CLASSES[i % len(CLASSES)]} for i in range(900)]
    df_dummy = pd.DataFrame(dummy_rows)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
        df_dummy.to_csv(tmp_path, index=False)

    try:
        validate_submission(tmp_path, expected_rows=900)
        print("[PASS] dummy submission passes validate_submission")
    finally:
        tmp_path.unlink()

    # also check that a wrong row count raises correctly
    df_bad = df_dummy.iloc[:50]
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_bad = Path(f.name)
        df_bad.to_csv(tmp_bad, index=False)
    try:
        validate_submission(tmp_bad, expected_rows=900)
        print("[FAIL] should have raised for wrong row count")
    except ValueError as e:
        print(f"[PASS] correctly raised for wrong row count: {e}")
    finally:
        tmp_bad.unlink()

    print("All submission tests passed.")
