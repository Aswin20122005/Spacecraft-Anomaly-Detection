from spacecraft_ad.synthetic import ANOMALY_KINDS, generate_telemetry


def test_shapes_and_index():
    tel, labels, _ = generate_telemetry(n_samples=2000, n_channels=3, n_anomalies=4, seed=1)
    assert tel.shape == (2000, 3)
    assert (tel.index == labels.index).all()
    assert str(tel.index.tz) == "UTC"


def test_anomalies_are_rare_and_labelled():
    _, labels, segments = generate_telemetry(seed=2)
    assert 0 < labels.mean() < 0.1
    for seg in segments:
        assert labels.iloc[seg["start"]:seg["end"]].all()


def test_training_quarter_is_clean():
    _, labels, _ = generate_telemetry(n_samples=4000, seed=3)
    assert labels.iloc[:1000].sum() == 0


def test_all_anomaly_kinds_are_used():
    _, _, segments = generate_telemetry(n_anomalies=8, seed=4)
    assert {s["kind"] for s in segments} == set(ANOMALY_KINDS)


def test_seed_makes_data_reproducible():
    a, _, _ = generate_telemetry(n_samples=500, n_anomalies=2, max_anomaly_len=50, seed=7)
    b, _, _ = generate_telemetry(n_samples=500, n_anomalies=2, max_anomaly_len=50, seed=7)
    assert a.equals(b)


def test_stuck_sensor_is_flat():
    tel, _, segments = generate_telemetry(seed=5)
    stuck = next(s for s in segments if s["kind"] == "stuck")
    assert tel[stuck["channel"]].iloc[stuck["start"]:stuck["end"]].nunique() == 1
