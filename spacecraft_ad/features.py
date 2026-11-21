"""Turn raw telemetry into per-sample features that carry local context.

Looking at single samples (as the notebook did) can't catch a stuck sensor or a slow
drift: each value on its own looks normal. Rolling statistics over a short window
expose those patterns.
"""

import pandas as pd


def window_features(telemetry, window=15):
    """For every channel add rolling mean, rolling std, first difference and the
    deviation from a longer baseline. Returns a DataFrame aligned with the input."""
    feats = {}
    for col in telemetry.columns:
        x = telemetry[col]
        roll = x.rolling(window, min_periods=1)
        baseline = x.rolling(window * 8, min_periods=1).median()
        feats[f"{col}_value"] = x
        feats[f"{col}_mean"] = roll.mean()
        feats[f"{col}_std"] = roll.std().fillna(0.0)
        feats[f"{col}_diff"] = x.diff().fillna(0.0)
        feats[f"{col}_dev"] = x - baseline
    return pd.DataFrame(feats, index=telemetry.index)
