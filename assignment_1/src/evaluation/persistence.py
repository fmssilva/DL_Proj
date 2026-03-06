# Saving and loading experiment results to/from disk.
# All functions are pure I/O — no model loading, no training logic here.
# Called by run_experiment (after each run) and by the final save-all cell.

import json
from pathlib import Path
from typing import Optional

from sklearn.metrics import f1_score, classification_report

from ..config import CLASSES


def save_experiment_result(name: str, entry: dict, results_path: Path) -> None:
    """
    Upsert one experiment entry into the results JSON on disk.
    entry is the same dict stored in results_tracker[name].
    Creates the file if it doesn't exist yet; merges if it does.
    history is serialised as-is (lists of floats) — readable and compact.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # load existing data so we never overwrite other experiments
    existing = {}
    if results_path.exists():
        with open(results_path) as f:
            existing = json.load(f)

    existing_exps = existing.get("experiments", {})
    existing_exps[name] = _serialise_entry(entry)
    existing["experiments"] = existing_exps

    with open(results_path, "w") as f:
        json.dump(existing, f, indent=2)


def save_all_results(
    results_tracker: dict,
    best_name: str,
    val_metrics: dict,
    all_labels: list,
    all_preds: list,
    config: dict,
    results_path: Path,
    classes: list = CLASSES,
) -> None:
    """
    Write the full results JSON for this task.
    Replaces the entire file — call this once after all experiments are done.

    val_metrics: {loss, acc, macro_f1} from evaluate() on the best model
    all_labels / all_preds: collected from the best model's val pass
    config: dict of run-level hyperparams (FAST_RUN, EPOCHS, LR, ...)
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # per-class F1 for the best model
    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = {cls: round(float(per_class_f1_arr[i]), 4) for i, cls in enumerate(classes)}

    # per-class classification report for the best model (string)
    cls_report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        zero_division=0,
    )

    # total GPU/CPU training time across non-ensemble experiments
    total_time = sum(
        v["train_time_s"] for v in results_tracker.values()
        if v.get("train_time_s", 0) > 0
    )

    output = {
        "best_experiment": best_name,
        "best_val_macro_f1": round(val_metrics["macro_f1"], 4),
        "best_val_accuracy": round(val_metrics["acc"], 4),
        "best_val_loss":     round(val_metrics["loss"], 4),
        "best_per_class_f1": per_class_f1,
        "best_classification_report": cls_report,
        "total_experiment_time_s": round(total_time, 1),
        "config": config,
        # full detail for every experiment — history included for trained models
        "experiments": {
            name: _serialise_entry(entry)
            for name, entry in results_tracker.items()
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved -> {results_path}")
    print(f"Best: {best_name}  val_macro_f1={output['best_val_macro_f1']}  val_acc={output['best_val_accuracy']}")
    print(f"Total experiment time: {total_time:.0f}s across {len(results_tracker)} experiments")


def load_results(results_path: Path) -> dict:
    """Load results JSON from disk — returns {} if file doesn't exist yet."""
    results_path = Path(results_path)
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return json.load(f)


def save_to_drive(
    task_out_dir: Path,
    task_name: str,
    root: str,
    in_colab: bool,
    drive_dir: str = "/content/drive/MyDrive/DL_Proj_Outputs",
) -> None:
    """
    Zip task_out_dir and copy the zip to Google Drive (Colab only).
    Zip is named '{task_name}_outputs.zip' — e.g. 'task1_outputs.zip'.
    Mounts Drive if not already mounted. On local runs just prints the path.
    """
    if not in_colab:
        print(f"[save_to_drive] local run — outputs at: {Path(task_out_dir).resolve()}")
        return

    import shutil
    from google.colab import drive as _drive  # type: ignore

    # mount Drive if the mount point doesn't have content yet
    drive_root = Path("/content/drive")
    if not (drive_root / "MyDrive").exists():
        _drive.mount(str(drive_root))

    # create the destination folder if needed
    dest_dir = Path(drive_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # build the zip in /tmp to avoid polluting the Colab workspace
    zip_stem = f"{task_name}_outputs"
    zip_path = Path("/tmp") / zip_stem
    shutil.make_archive(
        str(zip_path), "zip",
        root_dir=str(Path(task_out_dir).parent),
        base_dir=Path(task_out_dir).name,
    )
    final_zip = zip_path.with_suffix(".zip")

    # copy to Drive (overwrites if already exists — we always want the latest)
    dest_path = dest_dir / final_zip.name
    shutil.copy2(str(final_zip), str(dest_path))
    final_zip.unlink(missing_ok=True)   # clean up tmp zip

    print(f"[save_to_drive] saved -> {dest_path}")


def restore_from_drive(
    task_out_dir: Path,
    task_name: str,
    root: str,
    in_colab: bool,
    drive_dir: str = "/content/drive/MyDrive/DL_Proj_Outputs",
) -> bool:
    """
    Download and extract '{task_name}_outputs.zip' from Google Drive (Colab only).
    Extracts into task_out_dir's parent so the folder structure is restored intact.
    Returns True if a zip was found and extracted, False otherwise.
    On local runs always returns False — safe to call anywhere.
    """
    if not in_colab:
        print(f"[restore_from_drive] local run — skipping Drive restore.")
        return False

    import shutil
    from google.colab import drive as _drive  # type: ignore

    drive_root = Path("/content/drive")
    if not (drive_root / "MyDrive").exists():
        _drive.mount(str(drive_root))

    zip_name = f"{task_name}_outputs.zip"
    src_path = Path(drive_dir) / zip_name

    if not src_path.exists():
        print(f"[restore_from_drive] no '{zip_name}' found in Drive ({drive_dir}) — starting fresh.")
        return False

    # copy to /tmp first (faster than extracting directly from mounted Drive)
    tmp_zip = Path("/tmp") / zip_name
    shutil.copy2(str(src_path), str(tmp_zip))

    import zipfile
    extract_target = Path(task_out_dir).parent
    with zipfile.ZipFile(str(tmp_zip), "r") as zf:
        zf.extractall(str(extract_target))
    tmp_zip.unlink(missing_ok=True)

    print(f"[restore_from_drive] restored '{zip_name}' -> {extract_target}")
    return True


# ── internal helper ───────────────────────────────────────────────────────────

def _serialise_entry(entry: dict) -> dict:
    """
    Prepare one results_tracker entry for JSON.
    Rounds floats, keeps history lists, adds per_class_f1 if present.
    nan values (ensemble val_loss) are stored as null.
    """
    import math
    out = {}
    for k, v in entry.items():
        if isinstance(v, float):
            out[k] = None if math.isnan(v) else round(v, 4)
        elif isinstance(v, dict):
            # history dict: {"train_loss": [...], ...} or per_class_f1, child_models
            out[k] = {
                hk: [round(x, 4) for x in hv] if isinstance(hv, list) else hv
                for hk, hv in v.items()
            }
        else:
            out[k] = v
    return out


# ── local smoke tests ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.config import CLASSES

    tmp = Path(tempfile.mkdtemp())
    results_path = tmp / "results" / "test_results.json"

    # --- test save_experiment_result: create + upsert ---
    entry_A = {
        "val_macro_f1": 0.2373,
        "val_acc": 0.3012,
        "val_loss": 1.8432,
        "total_epochs": 12,
        "train_time_s": 87.4,
        "history": {"train_loss": [2.1, 1.9], "val_loss": [2.2, 2.0],
                    "train_f1": [0.10, 0.15], "val_f1": [0.09, 0.14]},
    }
    save_experiment_result("C_ls01_drop03", entry_A, results_path)
    data = load_results(results_path)
    assert "C_ls01_drop03" in data["experiments"]
    assert data["experiments"]["C_ls01_drop03"]["val_macro_f1"] == 0.2373
    print("[PASS] save_experiment_result creates file and stores entry correctly")

    # upsert a second experiment without losing the first
    entry_E = {**entry_A, "val_macro_f1": 0.2301, "train_time_s": 92.1}
    save_experiment_result("E_sampler", entry_E, results_path)
    data = load_results(results_path)
    assert "C_ls01_drop03" in data["experiments"]
    assert "E_sampler" in data["experiments"]
    print("[PASS] save_experiment_result upserts without losing previous entries")

    # --- test ensemble entry with nan val_loss ---
    entry_ens = {
        "val_macro_f1": 0.2450,
        "val_acc": 0.3100,
        "val_loss": float("nan"),
        "total_epochs": 0,
        "train_time_s": 0.0,
        "history": {},
        "child_models": ["C_ls01_drop03", "E_sampler"],
    }
    save_experiment_result("ENS_C_E", entry_ens, results_path)
    data = load_results(results_path)
    assert data["experiments"]["ENS_C_E"]["val_loss"] is None  # nan -> null in JSON
    assert data["experiments"]["ENS_C_E"]["child_models"] == ["C_ls01_drop03", "E_sampler"]
    print("[PASS] ensemble entry with nan val_loss serialised as null")

    # --- test save_all_results ---
    tracker = {"C_ls01_drop03": entry_A, "E_sampler": entry_E, "ENS_C_E": entry_ens}
    n_classes = len(CLASSES)
    dummy_labels = [i % n_classes for i in range(45)]
    dummy_preds  = [(i + 1) % n_classes for i in range(45)]
    val_metrics  = {"macro_f1": 0.2373, "acc": 0.3012, "loss": 1.8432}
    config       = {"FAST_RUN": True, "EPOCHS": 2, "LR": 0.001}

    all_results_path = tmp / "results" / "task_results.json"
    save_all_results(tracker, "C_ls01_drop03", val_metrics, dummy_labels, dummy_preds, config, all_results_path)
    data = load_results(all_results_path)
    assert data["best_experiment"] == "C_ls01_drop03"
    assert len(data["experiments"]) == 3
    assert len(data["best_per_class_f1"]) == n_classes
    assert "best_classification_report" in data
    print("[PASS] save_all_results wrote all 3 experiments with per-class F1 and report")

    # --- test load_results on missing file ---
    missing = load_results(tmp / "does_not_exist.json")
    assert missing == {}
    print("[PASS] load_results returns empty dict for missing file")

    # --- test save_to_drive / restore_from_drive non-Colab path ---
    save_to_drive(tmp, "task1", str(tmp), in_colab=False)
    result = restore_from_drive(tmp, "task1", str(tmp), in_colab=False)
    assert result is False
    print("[PASS] save_to_drive / restore_from_drive are safe no-ops when not on Colab")

    print("\nAll persistence.py smoke tests passed.")
