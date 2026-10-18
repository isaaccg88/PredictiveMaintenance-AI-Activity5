"""Evaluación de clasificadores y optimización de umbral SOLO en validación."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, fbeta_score, precision_score, recall_score,
    roc_auc_score,
)


def scores(y_true, proba, threshold: float = 0.5):
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "average_precision": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "threshold": float(threshold),
    }
    return result


def select_threshold(y_true, probabilities, min_precision=0.50):
    candidates = [scores(y_true, probabilities, float(t))
                  for t in np.round(np.arange(0.05, 0.951, 0.05), 2)]
    valid = [c for c in candidates if c["precision"] >= min_precision]
    # El criterio mínimo es duro cuando es factible; si ninguno lo cumple
    # se documenta el incumplimiento, sin fingir que cumple el requisito.
    feasible = len(valid) > 0
    pool = valid if feasible else candidates
    chosen = max(pool, key=lambda d: (d["f2"], d["recall"], d["precision"]))
    return chosen, candidates, feasible
