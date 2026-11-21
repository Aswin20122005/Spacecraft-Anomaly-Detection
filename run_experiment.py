"""Run both detectors on telemetry and print a results table.

    python run_experiment.py                      # synthetic telemetry
    python run_experiment.py --plot results/plot.png
"""

import argparse
import os

import pandas as pd

from spacecraft_ad.detectors import AutoencoderDetector, IsolationForestDetector, threshold_from_training
from spacecraft_ad.evaluate import evaluate
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
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--plot", help="save a figure of telemetry and scores to this path")
    args = parser.parse_args(argv)

    telemetry, labels, _ = generate_telemetry(n_samples=args.samples, seed=args.seed)
    results, scores, thresholds = run(telemetry, labels, window=args.window)
    print(f"Anomalous samples: {labels.mean():.1%}")
    print(results.to_string())
    if args.plot:
        os.makedirs(os.path.dirname(args.plot) or ".", exist_ok=True)
        print(f"Saved plot to {plot_results(telemetry, labels, scores, thresholds, args.plot)}")
    return results


if __name__ == "__main__":
    main()
