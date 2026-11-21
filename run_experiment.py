"""Run both detectors on telemetry and print a results table.

    python run_experiment.py                      # synthetic telemetry
    python run_experiment.py --plot results/plot.png
"""

import argparse

import pandas as pd

from spacecraft_ad.detectors import AutoencoderDetector, IsolationForestDetector, threshold_from_training
from spacecraft_ad.evaluate import evaluate
from spacecraft_ad.features import window_features
from spacecraft_ad.synthetic import generate_telemetry


def run(telemetry, labels, train_fraction=0.25, window=15, quantile=0.995):
    """Fit on the first part of the series (assumed normal), score everything,
    and evaluate on the rest. Returns (results DataFrame, scores dict, thresholds dict)."""
    X = window_features(telemetry, window=window)
    split = int(len(X) * train_fraction)
    X_train, X_test, y_test = X.iloc[:split], X.iloc[split:], labels.iloc[split:]

    rows, scores, thresholds = [], {}, {}
    for detector in (IsolationForestDetector(), AutoencoderDetector()):
        detector.fit(X_train)
        threshold = threshold_from_training(detector.score(X_train), quantile)
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
    args = parser.parse_args(argv)

    telemetry, labels, _ = generate_telemetry(n_samples=args.samples, seed=args.seed)
    results, _, _ = run(telemetry, labels, window=args.window)
    print(f"Anomalous samples: {labels.mean():.1%}")
    print(results.to_string())
    return results


if __name__ == "__main__":
    main()
