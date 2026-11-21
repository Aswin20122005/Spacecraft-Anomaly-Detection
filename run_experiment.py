"""Run both detectors on telemetry and print a results table.

    python run_experiment.py                      # synthetic telemetry
    python run_experiment.py --plot results/plot.png
    python run_experiment.py --esa data/ESA-Mission1 --channels channel_12 channel_13
"""

import argparse
import json
import os

import pandas as pd

from spacecraft_ad.detectors import AutoencoderDetector, IsolationForestDetector, threshold_from_training
from spacecraft_ad.esa import load_mission
from spacecraft_ad.evaluate import evaluate, recall_by_kind
from spacecraft_ad.features import window_features
from spacecraft_ad.plots import plot_results
from spacecraft_ad.synthetic import generate_telemetry


def run(telemetry, labels, train_fraction=0.25, window=15, period=95, quantile=0.995):
    """Fit on the first part of the series (assumed normal), calibrate the threshold on
    the end of that part, and evaluate on the rest. Returns (results DataFrame, scores dict, thresholds dict)."""
    X = window_features(telemetry, window=window, period=period)
    split = int(len(X) * train_fraction)
    fit_end = int(split * 0.8)
    X_fit, X_val = X.iloc[:fit_end], X.iloc[fit_end:split]
    X_test, y_test = X.iloc[split:], labels.iloc[split:]

    rows, scores, thresholds = [], {}, {}
    for detector in (IsolationForestDetector(), AutoencoderDetector()):
        detector.fit(X_fit)
        # Threshold from a held-out slice of the normal period: scores on the data a
        # model was fitted on are optimistically low, which made thresholds too tight.
        threshold = threshold_from_training(detector.score(X_val), quantile)
        test_scores = detector.score(X_test)
        rows.append({"model": detector.name, **evaluate(y_test, test_scores, threshold)})
        scores[detector.name] = pd.Series(test_scores, index=X_test.index)
        thresholds[detector.name] = threshold
    return pd.DataFrame(rows).set_index("model"), scores, thresholds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="average synthetic results over this many seeds")
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--plot", help="save a figure of telemetry and scores to this path")
    parser.add_argument("--out", help="save metrics as JSON to this path (for tracking results over time)")
    parser.add_argument("--esa", metavar="MISSION_DIR", help="use real ESA-ADB data from this folder")
    parser.add_argument("--channels", nargs="+", default=["channel_12"], help="ESA channels to use")
    parser.add_argument("--period", type=int, default=95, help="samples per orbit (default 95)")
    args = parser.parse_args(argv)

    segments = None
    by_kind = None
    if args.esa:
        telemetry, labels = load_mission(args.esa, args.channels)
    else:
        telemetry, labels, segments = generate_telemetry(n_samples=args.samples, seed=args.seed)
    results, scores, thresholds = run(telemetry, labels, window=args.window, period=args.period)

    print(f"Anomalous samples: {labels.mean():.1%}")
    print(results.to_string())
    if segments:
        offset = len(telemetry) - len(next(iter(scores.values())))
        by_kind = pd.DataFrame({
            name: recall_by_kind((s.to_numpy() > thresholds[name]).astype(int), segments, offset)
            for name, s in scores.items()
        }).T
        print("\nEvents caught by anomaly kind:")
        print(by_kind.to_string())
    if args.repeats > 1 and not args.esa:
        # One synthetic series can be lucky or unlucky; report mean and std over seeds.
        runs = [results] + [
            run(*generate_telemetry(n_samples=args.samples, seed=args.seed + k)[:2], window=args.window,
                period=args.period)[0]
            for k in range(1, args.repeats)
        ]
        stacked = pd.concat(runs, keys=range(len(runs)))
        mean, std = stacked.groupby(level=1).mean(), stacked.groupby(level=1).std()
        print(f"Mean over {args.repeats} seeds (± std):")
        print((mean.round(3).astype(str) + " ± " + std.round(3).astype(str)).to_string())
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        payload = {"source": args.esa or "synthetic", "seed": args.seed, "window": args.window,
                   "anomaly_rate": round(float(labels.mean()), 4), "results": results.to_dict(orient="index")}
        if by_kind is not None:
            payload["events_caught_by_kind"] = by_kind.to_dict(orient="index")
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {args.out}")
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        print(f"Saved plot to {plot_results(telemetry, labels, scores, thresholds, args.plot)}")
    return results


if __name__ == "__main__":
    main()
