import numpy as np
import pandas as pd
import pytest

from spacecraft_ad.detectors import AutoencoderDetector, IsolationForestDetector, threshold_from_training
from spacecraft_ad.features import window_features
from spacecraft_ad.synthetic import generate_telemetry


def test_features_have_no_missing_values():
    tel, _, _ = generate_telemetry(n_samples=1000, n_anomalies=2, max_anomaly_len=50)
    X = window_features(tel)
    assert X.shape == (1000, 5 * tel.shape[1])
    assert not X.isna().any().any()


def test_features_ignore_a_constant_offset():
    tel, _, _ = generate_telemetry(n_samples=1000, n_anomalies=2, max_anomaly_len=50)
    a = window_features(tel).iloc[400:]
    b = window_features(tel + 100).iloc[400:]
    pd.testing.assert_frame_equal(a, b, atol=1e-9)


def test_stuck_sensor_shows_zero_rolling_std():
    x = pd.Series(np.sin(np.arange(300) / 5.0))
    x.iloc[200:260] = x.iloc[200]
    feats = window_features(pd.DataFrame({"c": x}), window=10)
    assert feats["c_std"].iloc[220:260].max() == pytest.approx(0, abs=1e-12)
    assert feats["c_std"].iloc[50:150].min() > 0.05


@pytest.mark.parametrize("detector_cls", [IsolationForestDetector, AutoencoderDetector])
def test_detectors_score_outliers_higher(detector_cls):
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, (800, 4))
    outliers = rng.normal(8, 1, (20, 4))
    det = detector_cls().fit(normal)
    assert det.score(outliers).mean() > det.score(normal).mean()


def test_threshold_is_a_high_quantile():
    scores = np.arange(1000)
    assert threshold_from_training(scores, 0.99) == pytest.approx(989.01)
