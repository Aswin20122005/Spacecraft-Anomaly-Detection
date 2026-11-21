import numpy as np
import pytest

from spacecraft_ad.evaluate import evaluate, event_recall, point_adjust, segments


def test_segments_finds_runs_of_ones():
    assert segments([0, 1, 1, 0, 0, 1, 0, 1]) == [(1, 3), (5, 6), (7, 8)]
    assert segments([0, 0]) == []


def test_point_adjust_fills_detected_segments_only():
    y_true = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    assert point_adjust(y_true, y_pred).tolist() == [0, 1, 1, 1, 0, 0, 0, 1]


def test_event_recall_counts_segments_with_any_alarm():
    y_true = [0, 1, 1, 0, 1, 1, 0]
    assert event_recall(y_true, [0, 0, 1, 0, 0, 0, 0]) == 0.5
    assert np.isnan(event_recall([0, 0], [0, 1]))


def test_perfect_scores_give_perfect_metrics():
    y = np.array([0] * 90 + [1] * 10)
    m = evaluate(y, scores=y.astype(float), threshold=0.5)
    assert m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == m["pr_auc"] == 1.0
    assert m["alarm_rate"] == pytest.approx(0.1)


def test_random_scores_have_low_pr_auc():
    rng = np.random.default_rng(0)
    y = np.array([0] * 950 + [1] * 50)
    m = evaluate(y, rng.random(1000), threshold=0.99)
    assert m["pr_auc"] < 0.2
