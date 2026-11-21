"""Generate realistic multi-channel spacecraft telemetry with labelled anomaly segments.

Real telemetry is periodic (orbits), slightly noisy and correlated across channels.
Anomalies are rare (a few percent of samples) and come in a few typical shapes:
spikes, level shifts, drifts and frozen ("stuck") sensors.
"""

import numpy as np
import pandas as pd

ANOMALY_KINDS = ("spike", "level_shift", "drift", "stuck")


def generate_telemetry(
    n_samples=20_000,
    n_channels=4,
    n_anomalies=12,
    max_anomaly_len=120,
    orbit_period=95,  # minutes, like a low Earth orbit
    noise=0.05,
    seed=42,
):
    """Return (telemetry DataFrame, labels Series, segments list).

    telemetry: one column per channel, indexed by 1-minute UTC timestamps
    labels:    1 for anomalous samples, 0 otherwise
    segments:  list of dicts (start, end, kind, channel)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)
    index = pd.date_range("2004-12-01", periods=n_samples, freq="1min", tz="UTC")

    base = np.sin(2 * np.pi * t / orbit_period)
    data = {}
    for c in range(n_channels):
        phase = rng.uniform(0, 2 * np.pi)
        harmonic = 0.3 * np.sin(4 * np.pi * t / orbit_period + phase)
        trend = 0.0002 * t * rng.uniform(-1, 1)
        data[f"channel_{c + 1}"] = (
            rng.uniform(0.5, 2.0) * base + harmonic + trend + rng.normal(0, noise, n_samples)
        )
    telemetry = pd.DataFrame(data, index=index)
    labels = pd.Series(0, index=index, name="is_anomaly")

    segments = []
    # Keep anomalies out of the first 25% so it can be used as (mostly) clean training data.
    starts = np.sort(rng.choice(np.arange(n_samples // 4, n_samples - max_anomaly_len), n_anomalies, replace=False))
    for start in starts:
        length = int(rng.integers(10, max_anomaly_len))
        end = start + length
        kind = ANOMALY_KINDS[len(segments) % len(ANOMALY_KINDS)]
        col = telemetry.columns[int(rng.integers(n_channels))]
        values = telemetry[col].to_numpy().copy()
        seg = slice(start, end)
        if kind == "spike":
            values[seg] += rng.choice([-1, 1]) * rng.uniform(3, 5)
        elif kind == "level_shift":
            values[seg] += rng.choice([-1, 1]) * rng.uniform(1.5, 2.5)
        elif kind == "drift":
            values[seg] += np.linspace(0, rng.choice([-1, 1]) * 3, length)
        elif kind == "stuck":
            values[seg] = values[start]
        telemetry[col] = values
        labels.iloc[seg] = 1
        segments.append({"start": int(start), "end": int(end), "kind": kind, "channel": col})

    return telemetry, labels, segments
