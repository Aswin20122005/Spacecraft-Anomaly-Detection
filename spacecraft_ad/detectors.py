"""Unsupervised anomaly detectors that output a score (higher = more anomalous)."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class IsolationForestDetector:
    """Isolation Forest on scaled features."""

    name = "Isolation Forest"

    def __init__(self, n_estimators=200, random_state=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X):
        self.model.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X):
        # decision_function is high for normal points, so negate it.
        return -self.model.decision_function(self.scaler.transform(X))


class AutoencoderDetector:
    """A small bottleneck MLP trained to reconstruct normal data; the score is the
    reconstruction error. Uses scikit-learn, so no TensorFlow dependency."""

    name = "Autoencoder"

    def __init__(self, hidden=(16, 4, 16), max_iter=300, random_state=42):
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden, max_iter=max_iter, early_stopping=True, random_state=random_state
        )

    def fit(self, X):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, Xs)
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return np.mean((Xs - self.model.predict(Xs)) ** 2, axis=1)


def threshold_from_training(train_scores, quantile=0.995):
    """Pick the alarm threshold from scores on (normal) training data, not from the
    labels. The notebook set it using the true anomaly rate, which leaks the answer."""
    return float(np.quantile(train_scores, quantile))
