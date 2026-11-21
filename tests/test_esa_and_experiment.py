import pickle
import zipfile

import numpy as np
import pandas as pd

import run_experiment
from spacecraft_ad.esa import load_channel, load_mission


def make_fake_mission(root):
    """Write a tiny dataset in the ESA-ADB layout."""
    (root / "channels").mkdir(parents=True)
    index = pd.date_range("2004-12-01", periods=600, freq="30s", tz="UTC")
    for i, name in enumerate(["channel_1", "channel_2"]):
        df = pd.DataFrame({name: np.sin(np.arange(600) / 10 + i)}, index=index)
        with zipfile.ZipFile(root / "channels" / f"{name}.zip", "w") as z:
            z.writestr(name, pickle.dumps(df))
    pd.DataFrame(
        {
            "ID": ["id_1", "id_2"],
            "Channel": ["channel_1", "channel_9"],
            "StartTime": ["2004-12-01T01:00:00Z", "2004-12-01T02:00:00Z"],
            "EndTime": ["2004-12-01T01:10:00Z", "2004-12-01T02:10:00Z"],
        }
    ).to_csv(root / "labels.csv", index=False)


def test_load_channel_reads_zipped_pickle(tmp_path):
    make_fake_mission(tmp_path)
    s = load_channel(tmp_path, "channel_1")
    assert len(s) == 600 and str(s.index.tz) == "UTC"


def test_load_mission_resamples_and_labels_selected_channels(tmp_path):
    make_fake_mission(tmp_path)
    tel, labels = load_mission(tmp_path, ["channel_1", "channel_2"])
    assert list(tel.columns) == ["channel_1", "channel_2"]
    assert tel.index.freq == "min" or (tel.index[1] - tel.index[0]) == pd.Timedelta("1min")
    assert labels.sum() == 11  # 01:00..01:10 inclusive; channel_9's window is ignored


def test_experiment_runs_end_to_end_on_synthetic_data():
    results = run_experiment.main([])
    assert set(results.index) == {"Isolation Forest", "Autoencoder"}
    assert (results["roc_auc"] > 0.85).all()
    assert (results["event_recall"] >= 0.9).all()
    assert (results["alarm_rate"] < 0.15).all()


def test_experiment_saves_plot(tmp_path):
    out = tmp_path / "plot.png"
    run_experiment.main(["--samples", "4000", "--plot", str(out)])
    assert out.stat().st_size > 10_000


def test_experiment_saves_metrics_json(tmp_path):
    import json

    out = tmp_path / "metrics.json"
    run_experiment.main(["--samples", "4000", "--out", str(out)])
    data = json.loads(out.read_text())
    assert data["source"] == "synthetic"
    assert set(data["results"]) == {"Isolation Forest", "Autoencoder"}
    assert 0 <= data["results"]["Autoencoder"]["pr_auc"] <= 1
    assert set(data["events_caught_by_kind"]["Autoencoder"]) <= {"spike", "level_shift", "drift", "stuck"}
