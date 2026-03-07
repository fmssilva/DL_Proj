# Pure I/O for experiment results and Drive backups.
# No model loading, no training logic here.
# Called by run_experiment (after each run) and by the notebook's save/restore cells.

import json
from pathlib import Path
from typing import TypedDict

from sklearn.metrics import f1_score, classification_report

from ..config import CLASSES


# ══════════════════════════════════════════════════════════════════════════════
# Experiment result schema
# ══════════════════════════════════════════════════════════════════════════════

class ExperimentEntry(TypedDict):
    """
    One row in results_tracker — produced by run_experiment (solo) or the ensemble cell.

    Fields set by run_experiment (solo models):
        val_macro_f1 : best val macro-F1 from the saved checkpoint
        val_acc      : val accuracy at the same checkpoint
        val_loss     : val loss at the same checkpoint
        total_epochs : how many epochs actually ran (may be < EPOCHS if early-stopped)
        train_time_s : wall-clock seconds for the full training loop
        history      : {"train_loss": [...], "val_loss": [...],
                        "train_f1": [...], "val_f1": [...]}  -- one float per epoch

    Extra field set only by ensemble cells:
        child_models : list of experiment names that were averaged (e.g. ["C_ls01_drop03", "E_sampler"])
                       absent on solo entries, val_loss=nan and total_epochs=0 for ensembles
    """
    val_macro_f1: float
    val_acc:      float
    val_loss:     float
    total_epochs: int
    train_time_s: float
    history:      dict   # keys: train_loss, val_loss, train_f1, val_f1 — each a list of floats


# shorthand used in function signatures throughout this file and analysis.py
ResultsTracker = dict[str, ExperimentEntry]


# ── private helper ─────────────────────────────────────────────────────────────

def _serialise_entry(entry: ExperimentEntry) -> dict:
    # prepare one results_tracker entry for JSON: round floats, nan -> null
    import math
    out = {}
    for k, v in entry.items():
        if isinstance(v, float):
            out[k] = None if math.isnan(v) else round(v, 4)
        elif isinstance(v, dict):
            out[k] = {
                hk: [round(x, 4) for x in hv] if isinstance(hv, list) else hv
                for hk, hv in v.items()
            }
        else:
            out[k] = v
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 1. RESULTS JSON  (read / write experiment results to disk)
# ══════════════════════════════════════════════════════════════════════════════

def save_experiment_result(name: str, entry: ExperimentEntry, results_path: Path) -> None:
    # upsert one experiment into the results JSON; never overwrites other experiments
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

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
    results_tracker: ResultsTracker,
    best_name: str,
    val_metrics: dict,
    all_labels: list,
    all_preds: list,
    config: dict,
    results_path: Path,
    classes: list = CLASSES,
) -> None:
    # write full results JSON for the task — call once after all experiments are done
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    per_class_f1_arr = f1_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = {cls: round(float(per_class_f1_arr[i]), 4) for i, cls in enumerate(classes)}

    cls_report = classification_report(
        all_labels, all_preds,
        target_names=classes,
        labels=list(range(len(classes))),
        zero_division=0,
    )

    total_time = sum(
        v["train_time_s"] for v in results_tracker.values()
        if v.get("train_time_s", 0) > 0
    )

    output = {
        "best_experiment":            best_name,
        "best_val_macro_f1":          round(val_metrics["macro_f1"], 4),
        "best_val_accuracy":          round(val_metrics["acc"], 4),
        "best_val_loss":              round(val_metrics["loss"], 4),
        "best_per_class_f1":          per_class_f1,
        "best_classification_report": cls_report,
        "total_experiment_time_s":    round(total_time, 1),
        "config": config,
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
    # load raw JSON from disk; returns {} if file doesn't exist yet
    results_path = Path(results_path)
    if not results_path.exists():
        return {}
    with open(results_path) as f:
        return json.load(f)


def restore_tracker(results_path: Path, tracker: ResultsTracker) -> None:
    """
    Load experiments from a results JSON into tracker (in-place).
    Supports both the current 'experiments' key and the old 'all_experiments' key.
    Prints how many experiments were loaded (or a 'starting fresh' message).
    Used by the notebook's Results Manager cell instead of the old inline _restore_from_json.
    """
    data = load_results(results_path)
    if not data:
        print(f"No results JSON found at {results_path} — starting fresh.")
        return
    # support both current and legacy key name
    exps = data.get("experiments") or data.get("all_experiments") or {}
    tracker.update(exps)
    print(f"Restored {len(exps)} experiments from {results_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. DRIVE PRIMITIVES  (download / extract from public Drive folder)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_folder_id(folder_url: str) -> str:
    # extract folder ID from any Google Drive folder share URL; raises ValueError if not a Drive link
    import re
    m = re.search(r"/folders/([a-zA-Z0-9_-]+)", folder_url)
    if not m:
        raise ValueError(
            f"Cannot parse folder ID from URL: {folder_url!r}\n"
            "Expected format: https://drive.google.com/drive/folders/<ID>"
        )
    return m.group(1)


def download_from_drive(folder_url: str, file_name: str, dst_path: str, quiet: bool = False) -> bool:
    """
    Download a single named file from a public shared Google Drive folder.

    folder_url : 'Anyone with the link' share URL of the Drive folder
    file_name  : exact file name inside that folder, e.g. 'data.zip'
    dst_path   : local destination path
    quiet      : suppress gdown progress bar

    Uses gdown folder listing (metadata only) to find the file by name,
    then downloads by file ID. Returns True on success, False on any failure.
    Works identically locally and on Colab — no google.colab import needed.
    """
    try:
        import gdown
    except ImportError:
        print("[download_from_drive] gdown not installed — run: pip install gdown")
        return False

    try:
        folder_id = _parse_folder_id(folder_url)
    except ValueError as e:
        print(f"[download_from_drive] {e}")
        return False

    # list folder contents — metadata only, no file download yet
    try:
        files = gdown.download_folder(
            id=folder_id,
            quiet=True,
            use_cookies=False,
            skip_download=True,
        )
    except Exception as e:
        print(f"[download_from_drive] failed to list folder '{folder_id}': {e}")
        return False

    if not files:
        print(f"[download_from_drive] folder '{folder_id}' is empty or not accessible.")
        return False

    # gdown >=5 returns objects with .path/.id; older versions return dicts
    target = None
    for f in files:
        name = getattr(f, "path", None) or getattr(f, "name", None)
        if name is None and isinstance(f, dict):
            name = f.get("name") or f.get("path")
        fid = getattr(f, "id", None)
        if fid is None and isinstance(f, dict):
            fid = f.get("id")
        if name and Path(name).name == file_name and fid:
            target = fid
            break

    if target is None:
        available = []
        for f in files:
            n = getattr(f, "path", None) or getattr(f, "name", None)
            if n is None and isinstance(f, dict):
                n = f.get("name") or f.get("path")
            available.append(Path(n).name if n else str(f))
        print(
            f"[download_from_drive] '{file_name}' not found in folder.\n"
            f"  Available: {available}"
        )
        return False

    try:
        result = gdown.download(id=target, output=str(dst_path), quiet=quiet)
        if result is None:
            print(f"[download_from_drive] gdown returned None for '{file_name}' (id={target!r}).")
            return False
        print(f"[download_from_drive] '{file_name}' -> {dst_path}")
        return True
    except Exception as e:
        print(f"[download_from_drive] download failed: {e}")
        return False


def extract_zip(zip_path: str, dst_dir: str, remove_zip: bool = True) -> None:
    # extract zip_path into dst_dir; creates dst_dir if needed; optionally removes the zip after
    import zipfile
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dst))
    if remove_zip:
        Path(zip_path).unlink(missing_ok=True)


def download_and_extract(folder_url: str, file_name: str, dst_dir: str, quiet: bool = False) -> bool:
    # download a named zip from a public Drive folder and extract into dst_dir; cleans up temp zip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_zip = str(Path(tmp) / file_name)
        ok = download_from_drive(folder_url, file_name, tmp_zip, quiet=quiet)
        if not ok:
            return False
        extract_zip(tmp_zip, dst_dir, remove_zip=False)  # temp dir handles cleanup
    return True


# ══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUTS BACKUP  (zip and save / restore the task outputs folder)
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    task_out_dir: Path,
    task_name: str,
    in_colab: bool,
    use_drive: bool,
    drive_dir: str,
) -> None:
    """
    Zip task_out_dir and save it to drive_dir as '{task_name}_outputs.zip'.

    in_colab=True  : mounts Google Drive and saves to the mounted path (drive_dir).
    in_colab=False : saves the zip directly to drive_dir as a local backup folder.
    use_drive=False: skips silently and prints the local outputs path.
    """
    if not use_drive:
        print(f"[save_outputs] USE_DRIVE=False — outputs at: {Path(task_out_dir).resolve()}")
        return

    import shutil

    if in_colab:
        from google.colab import drive as _drive  # type: ignore
        drive_root = Path("/content/drive")
        if not (drive_root / "MyDrive").exists():
            _drive.mount(str(drive_root))

    dest_dir = Path(drive_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_stem = f"{task_name}_outputs"
    # zip to /tmp on Colab to avoid slow direct writes to mounted Drive
    tmp_zip = Path("/tmp") / zip_stem if in_colab else dest_dir / zip_stem
    shutil.make_archive(
        str(tmp_zip), "zip",
        root_dir=str(Path(task_out_dir).parent),
        base_dir=Path(task_out_dir).name,
    )
    final_zip = tmp_zip.with_suffix(".zip")

    if in_colab:
        dest_path = dest_dir / final_zip.name
        shutil.copy2(str(final_zip), str(dest_path))
        final_zip.unlink(missing_ok=True)
        print(f"[save_outputs] saved -> {dest_path}")
    else:
        print(f"[save_outputs] saved -> {final_zip}")


def restore_outputs(
    task_out_dir: Path,
    task_name: str,
    in_colab: bool,
    backup_dir: str,
) -> bool:
    """
    Restore '{task_name}_outputs.zip' and extract it into task_out_dir's parent.

    in_colab=True  : backup_dir is a mounted Drive path (e.g. '/content/drive/MyDrive/DL_Proj').
                     Mounts Drive, copies zip to /tmp, extracts.
    in_colab=False : backup_dir is a public Drive folder URL.
                     Downloads via gdown, extracts.

    Returns True if found and extracted, False otherwise.
    Guard with 'if USE_DRIVE:' in the notebook before calling.
    """
    import shutil

    zip_name = f"{task_name}_outputs.zip"
    extract_target = Path(task_out_dir).parent

    if in_colab:
        from google.colab import drive as _drive  # type: ignore
        drive_root = Path("/content/drive")
        if not (drive_root / "MyDrive").exists():
            _drive.mount(str(drive_root))
        src_path = Path(backup_dir) / zip_name
        if not src_path.exists():
            print(f"[restore_outputs] '{zip_name}' not found in {backup_dir} — starting fresh.")
            return False
        tmp_zip = Path("/tmp") / zip_name
        shutil.copy2(str(src_path), str(tmp_zip))
        extract_zip(str(tmp_zip), str(extract_target), remove_zip=True)
    else:
        ok = download_and_extract(backup_dir, zip_name, str(extract_target))
        if not ok:
            print(f"[restore_outputs] '{zip_name}' not found in Drive folder — starting fresh.")
            return False

    print(f"[restore_outputs] restored '{zip_name}' -> {extract_target}")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL SMOKE TESTS
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from src.config import CLASSES

    tmp = Path(tempfile.mkdtemp())
    results_path = tmp / "results" / "test_results.json"

    # ── save_experiment_result: create ────────────────────────────────────────
    entry_A = {
        "val_macro_f1": 0.2373, "val_acc": 0.3012, "val_loss": 1.8432,
        "total_epochs": 12, "train_time_s": 87.4,
        "history": {"train_loss": [2.1, 1.9], "val_loss": [2.2, 2.0],
                    "train_f1": [0.10, 0.15], "val_f1": [0.09, 0.14]},
    }
    save_experiment_result("C_ls01_drop03", entry_A, results_path)
    data = load_results(results_path)
    assert "C_ls01_drop03" in data["experiments"]
    assert data["experiments"]["C_ls01_drop03"]["val_macro_f1"] == 0.2373
    print("[PASS] save_experiment_result creates file and stores entry")

    # ── save_experiment_result: upsert ────────────────────────────────────────
    entry_E = {**entry_A, "val_macro_f1": 0.2301, "train_time_s": 92.1}
    save_experiment_result("E_sampler", entry_E, results_path)
    data = load_results(results_path)
    assert "C_ls01_drop03" in data["experiments"] and "E_sampler" in data["experiments"]
    print("[PASS] save_experiment_result upserts without losing previous entries")

    # ── ensemble entry: nan val_loss -> null ──────────────────────────────────
    entry_ens = {
        "val_macro_f1": 0.2450, "val_acc": 0.3100, "val_loss": float("nan"),
        "total_epochs": 0, "train_time_s": 0.0, "history": {},
        "child_models": ["C_ls01_drop03", "E_sampler"],
    }
    save_experiment_result("ENS_C_E", entry_ens, results_path)
    data = load_results(results_path)
    assert data["experiments"]["ENS_C_E"]["val_loss"] is None
    assert data["experiments"]["ENS_C_E"]["child_models"] == ["C_ls01_drop03", "E_sampler"]
    print("[PASS] ensemble entry with nan val_loss serialised as null")

    # ── save_all_results ──────────────────────────────────────────────────────
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
    print("[PASS] save_all_results wrote all experiments with per-class F1 and report")

    # ── load_results: missing file ────────────────────────────────────────────
    assert load_results(tmp / "does_not_exist.json") == {}
    print("[PASS] load_results returns empty dict for missing file")

    # ── restore_tracker ───────────────────────────────────────────────────────
    fresh_tracker: dict = {}
    restore_tracker(all_results_path, fresh_tracker)
    assert set(fresh_tracker.keys()) == {"C_ls01_drop03", "E_sampler", "ENS_C_E"}
    assert fresh_tracker["C_ls01_drop03"]["val_macro_f1"] == 0.2373
    print("[PASS] restore_tracker loads all experiments into tracker dict")

    empty_tracker: dict = {}
    restore_tracker(tmp / "missing.json", empty_tracker)
    assert empty_tracker == {}
    print("[PASS] restore_tracker handles missing file gracefully")

    # ── save_outputs: creates zip ─────────────────────────────────────────────
    task_out = tmp / "task1"
    task_out.mkdir(parents=True, exist_ok=True)
    (task_out / "dummy.txt").write_text("hello")
    (task_out / "results").mkdir()
    (task_out / "results" / "task1_results.json").write_text('{"experiments":{}}')

    fake_drive = tmp / "fake_drive"
    save_outputs(task_out, "task1", in_colab=False, use_drive=True, drive_dir=str(fake_drive))
    zip_path = fake_drive / "task1_outputs.zip"
    assert zip_path.exists(), "zip not created by save_outputs"
    print("[PASS] save_outputs created zip in local backup folder")

    # ── save_outputs: zip contents are correct ────────────────────────────────
    import zipfile
    with zipfile.ZipFile(str(zip_path)) as zf:
        names = zf.namelist()
    assert any("dummy.txt" in n for n in names), f"dummy.txt not in zip: {names}"
    assert any("results" in n for n in names), f"results/ not in zip: {names}"
    print(f"[PASS] save_outputs zip contains expected files: {names}")

    # ── save_outputs: USE_DRIVE=False is silent no-op ─────────────────────────
    save_outputs(task_out, "task1", in_colab=False, use_drive=False, drive_dir=str(fake_drive))
    print("[PASS] save_outputs is a silent no-op when use_drive=False")

    # ── restore_outputs: round-trip from local zip ────────────────────────────
    # remove the output folder and re-extract from the zip we just created
    import shutil as _shutil
    _shutil.rmtree(str(task_out))
    assert not task_out.exists()
    # extract directly (simulating the non-Colab path with a local file)
    extract_zip(str(zip_path), str(task_out.parent), remove_zip=False)
    assert (task_out / "dummy.txt").exists(), "dummy.txt not restored after extract_zip"
    assert (task_out / "results" / "task1_results.json").exists()
    print("[PASS] round-trip: save_outputs zip re-extracted correctly")

    # ── restore_outputs: bad URL returns False ────────────────────────────────
    task_out_restore = tmp / "task1_restored"
    fake_url = "https://drive.google.com/drive/folders/FAKE_ID_FOR_LOCAL_TEST"
    ok = restore_outputs(task_out_restore, "task1", in_colab=False, backup_dir=fake_url)
    assert ok is False
    assert not task_out_restore.exists()
    print("[PASS] restore_outputs returns False for bad Drive URL")

    # ── restore_outputs: real folder, zip not present (network) ──────────────
    print("\n[INFO] Testing restore_outputs with real Drive folder (network) ...")
    _real_folder = "https://drive.google.com/drive/folders/1u2Xw2-4_L5OhPFjY_OsgNbMayrTI-ROR?usp=sharing"
    ok_real = restore_outputs(task_out_restore, "__nonexistent_task__", in_colab=False, backup_dir=_real_folder)
    assert ok_real is False
    print("[PASS] restore_outputs returns False when zip not in real Drive folder")

    # ── _parse_folder_id ─────────────────────────────────────────────────────
    FOLDER_URL = "https://drive.google.com/drive/folders/1u2Xw2-4_L5OhPFjY_OsgNbMayrTI-ROR?usp=sharing"
    assert _parse_folder_id(FOLDER_URL) == "1u2Xw2-4_L5OhPFjY_OsgNbMayrTI-ROR"
    assert _parse_folder_id("https://drive.google.com/drive/folders/ABC123xyz") == "ABC123xyz"
    print("[PASS] _parse_folder_id extracts ID from Drive folder URLs")
    try:
        _parse_folder_id("https://example.com/not-a-drive-link")
        assert False
    except ValueError:
        pass
    print("[PASS] _parse_folder_id raises ValueError on non-Drive URL")

    # ── download_from_drive: bad URL ──────────────────────────────────────────
    assert download_from_drive("not-a-url", "data.zip", "/tmp/test.zip", quiet=True) is False
    print("[PASS] download_from_drive returns False on malformed URL")

    # ── download_from_drive: real folder, missing file (network) ─────────────
    print("\n[INFO] Testing real Drive folder access (network) ...")
    ok_missing = download_from_drive(FOLDER_URL, "__no_such_file__.zip", "/tmp/no.zip", quiet=True)
    assert ok_missing is False
    assert not Path("/tmp/no.zip").exists()
    print("[PASS] download_from_drive returns False for file not in folder")

    ok_bad = download_from_drive(
        "https://drive.google.com/drive/folders/__bad_id__", "x.zip", "/tmp/x.zip", quiet=True
    )
    assert ok_bad is False
    print("[PASS] download_from_drive returns False for bad folder ID")

    # ── extract_zip ───────────────────────────────────────────────────────────
    zip_src_dir = tmp / "zip_source"
    zip_src_dir.mkdir()
    (zip_src_dir / "hello.txt").write_text("world")
    import shutil as _shutil2
    _shutil2.make_archive(str(tmp / "test_extract"), "zip",
                          root_dir=str(zip_src_dir.parent), base_dir=zip_src_dir.name)
    archive_path = tmp / "test_extract.zip"

    extract_zip(str(archive_path), str(tmp / "extracted"), remove_zip=False)
    assert (tmp / "extracted" / "zip_source" / "hello.txt").exists()
    assert archive_path.exists()
    print("[PASS] extract_zip extracts correctly, keeps zip when remove_zip=False")

    extract_zip(str(archive_path), str(tmp / "extracted2"), remove_zip=True)
    assert (tmp / "extracted2" / "zip_source" / "hello.txt").exists()
    assert not archive_path.exists()
    print("[PASS] extract_zip deletes zip when remove_zip=True")

    # ── download_and_extract ──────────────────────────────────────────────────
    assert download_and_extract("not-a-url", "data.zip", str(tmp / "bad"), quiet=True) is False
    print("[PASS] download_and_extract returns False on bad URL")

    print("\n[INFO] Testing download_and_extract real folder (network) ...")
    ok_dl = download_and_extract(FOLDER_URL, "__no_such_file__.zip", str(tmp / "dl"), quiet=True)
    assert ok_dl is False
    print("[PASS] download_and_extract returns False for file not in folder")

    print("\nAll persistence.py smoke tests passed.")
