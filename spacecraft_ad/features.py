"""Turn raw telemetry into per-sample features that carry local context.

Looking at single samples (as the notebook did) can't catch a stuck sensor or a slow
drift: each value on its own looks normal. Rolling statistics over a short window
expose those patterns.

Absolute levels are deliberately left out. Telemetry drifts slowly over a mission, so
a model trained on the first weeks would flag every later sample as unusual. All
features are relative: to the recent past, or to the same point one orbit earlier.
"""

import pandas as pd

DEFAULT_PERIOD = 95  # samples per orbit (minutes, for 1-minute telemetry in LEO)


def window_features(telemetry, window=15, period=DEFAULT_PERIOD):
    """Per channel: rolling std, first difference, deviation from a longer rolling
    median, difference from one orbit ago and its rolling mean."""
    feats = {}
    for col in telemetry.columns:
        x = telemetry[col]
        baseline = x.rolling(window * 8, min_periods=1).median()
        orbit_diff = (x - x.shift(period)).fillna(0.0)
        feats[f"{col}_std"] = x.rolling(window, min_periods=1).std().fillna(0.0)
        feats[f"{col}_diff"] = x.diff().fillna(0.0)
        feats[f"{col}_dev"] = x - baseline
        feats[f"{col}_orbit_diff"] = orbit_diff
        feats[f"{col}_orbit_mean"] = orbit_diff.rolling(window, min_periods=1).mean()
    return pd.DataFrame(feats, index=telemetry.index)
