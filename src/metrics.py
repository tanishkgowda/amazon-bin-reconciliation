"""Evaluation metrics.

Besides plain accuracy we report:
  * RMSE / MAE of the predicted count (an off-by-3 error is worse than off-by-1)
  * off-by-one accuracy (|pred - true| <= 1)
  * per-class accuracy and macro-F1 (classes are imbalanced)
  * a reconciliation metric: the real business task is "does this bin match what the
    inventory system says?", so we simulate recorded quantities (half correct, half wrong)
    and measure how well 1 - P(count == recorded) separates mismatches (ROC-AUC).
"""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .config import NUM_CLASSES, SEED


def compute_metrics(y_true, y_pred, probs=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    err = y_pred - y_true
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    per_class = {str(k): float(cm[k, k] / cm[k].sum()) if cm[k].sum() else None for k in range(NUM_CLASSES)}
    m = {
        "n": int(len(y_true)),
        "accuracy": float((err == 0).mean()),
        "off_by_one_accuracy": float((np.abs(err) <= 1).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }
    if probs is not None:
        probs = np.asarray(probs)
        expected = probs @ np.arange(NUM_CLASSES)
        m["rmse_expected_count"] = float(np.sqrt(((expected - y_true) ** 2).mean()))
        m["reconciliation"] = simulate_reconciliation(probs, y_true)
    return m


def simulate_reconciliation(probs, y_true, mismatch_rate=0.5, threshold=0.5, seed=SEED):
    """Simulate an inventory system whose recorded count is wrong for `mismatch_rate` of bins.

    Wrong records are mostly off by one (the realistic case: one item misplaced),
    sometimes a random other count. A bin is flagged when P(count == recorded) < threshold.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    is_mismatch = rng.random(n) < mismatch_rate
    recorded = y_true.copy()
    for i in np.where(is_mismatch)[0]:
        if rng.random() < 0.7:
            choices = [c for c in (y_true[i] - 1, y_true[i] + 1) if 0 <= c < NUM_CLASSES]
        else:
            choices = [c for c in range(NUM_CLASSES) if c != y_true[i]]
        recorded[i] = rng.choice(choices)
    p_match = probs[np.arange(n), recorded]
    score = 1.0 - p_match                    # higher = more likely a mismatch
    flagged = p_match < threshold
    return {
        "mismatch_rate": mismatch_rate,
        "roc_auc": float(roc_auc_score(is_mismatch, score)),
        "threshold": threshold,
        "precision": float(precision_score(is_mismatch, flagged, zero_division=0)),
        "recall": float(recall_score(is_mismatch, flagged, zero_division=0)),
        "flag_rate": float(flagged.mean()),
    }


def one_hot_probs(y_pred):
    """For models without probabilities (e.g. LinearSVC) fall back to one-hot 'probabilities'."""
    p = np.zeros((len(y_pred), NUM_CLASSES))
    p[np.arange(len(y_pred)), np.asarray(y_pred)] = 1.0
    return p
