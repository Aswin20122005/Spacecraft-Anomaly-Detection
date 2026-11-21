"""Load ESA Anomaly Detection Benchmark (ESA-ADB) mission data.

Download ESA-Mission1.zip from https://zenodo.org/records/12528696 and extract it, e.g.
to data/ESA-Mission1. Layout:

    ESA-Mission1/
        channels.csv          Channel, Subsystem, Physical Unit, Group, Target
        labels.csv            ID, Channel, StartTime, EndTime
        channels/channel_N.zip   each holds a pickled DataFrame with a datetime index

The notebook only used labels.csv and then generated fake telemetry; this module
loads the real channel signals and builds matching per-sample labels.
"""

import os
import pickle
import zipfile

import pandas as pd


def load_channel(mission_dir, channel):
    """Return one channel as a float Series indexed by UTC timestamps."""
    path = os.path.join(mission_dir, "channels", f"{channel}.zip")
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as f:
            obj = pickle.load(f)
    series = obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else pd.Series(obj)
    series.index = pd.to_datetime(series.index, utc=True)
    return series.astype(float).rename(channel)


def load_labels(mission_dir):
    labels = pd.read_csv(os.path.join(mission_dir, "labels.csv"))
    labels["StartTime"] = pd.to_datetime(labels["StartTime"], utc=True)
    labels["EndTime"] = pd.to_datetime(labels["EndTime"], utc=True)
    return labels


def load_mission(mission_dir, channels, start=None, end=None, freq="1min"):
    """Load several channels resampled onto a common grid, plus 0/1 anomaly labels.

    Returns (telemetry DataFrame, labels Series). A sample is anomalous if it falls in
    any labelled anomaly window for one of the selected channels.
    """
    series = [load_channel(mission_dir, c) for c in channels]
    telemetry = pd.concat(series, axis=1).resample(freq).mean().interpolate(limit_direction="both")
    if start is not None or end is not None:
        telemetry = telemetry.loc[start:end]

    windows = load_labels(mission_dir)
    windows = windows[windows["Channel"].isin(channels)]
    is_anomaly = pd.Series(0, index=telemetry.index, name="is_anomaly")
    for row in windows.itertuples():
        is_anomaly.loc[row.StartTime:row.EndTime] = 1
    return telemetry, is_anomaly
