# Evaluation metrics: macro F1 and per-class classification report.
# Imported by train.py and task entry-points.

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as sklearn_f1


def compute_macro_f1(y_true: list, y_pred: list) -> float:
    """Macro-averaged F1 score — the competition metric."""
    return sklearn_f1(y_true, y_pred, average="macro", zero_division=0)


def classification_report_str(y_true: list, y_pred: list, classes: list) -> str:
    """Per-class precision/recall/F1 as a formatted string for logging."""
    return classification_report(y_true, y_pred, target_names=classes, zero_division=0)
