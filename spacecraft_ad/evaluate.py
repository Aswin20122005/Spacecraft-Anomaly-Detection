"""Metrics for time-series anomaly detection."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def segments(labels):
    """Return [(start, end)] index ranges (end exclusive) where labels == 1."""
    y = np.asarray(labels).astype(int)
    edges = np.diff(np.concatenate([[0], y, [0]]))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def point_adjust(y_true, y_pred):
    """Standard 'point-adjusted' predictions: if any point of a true anomaly segment is
    flagged, the whole segment counts as detected. Operators care whether an event was
    caught, not whether every minute of it was flagged."""
    adjusted = np.asarray(y_pred).astype(int).copy()
    for start, end in segments(y_true):
        if adjusted[start:end].any():
            adjusted[start:end] = 1
    return adjusted


def event_recall(y_true, y_pred):
    """Fraction of anomaly segments with at least one alarm."""
    segs = segments(y_true)
    if not segs:
        return float("nan")
    y_pred = np.asarray(y_pred)
    return float(np.mean([y_pred[s:e].any() for s, e in segs]))


def evaluate(y_true, scores, threshold):
    """Point-wise and event-level metrics for one detector."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    pa_p, pa_r, pa_f1, _ = precision_recall_fscore_support(
        y_true, point_adjust(y_true, y_pred), average="binary", zero_division=0
    )
    return {
        "precision": round(float(p), 3),
        "recall": round(float(r), 3),
        "f1": round(float(f1), 3),
        "pa_f1": round(float(pa_f1), 3),
        "event_recall": round(event_recall(y_true, y_pred), 3),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 3),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 3),
        "alarm_rate": round(float(y_pred.mean()), 4),
    }
