"""AnomalyDetector wraps a Keras model and handles threshold calibration."""

import numpy as np


class AnomalyDetector:
    """Wraps a Keras binary classifier and finds an optimal decision threshold.

    The detector receives a trained model and does not build or train it.
    Threshold is calibrated on a validation set to maximize F1.

    Args:
        model: Trained tf.keras.Model with sigmoid output.
    """

    def __init__(self, model):
        self.model = model
        self.threshold = 0.5

    def fit_threshold(self, X_val, y_val):
        """Find the probability threshold that maximizes F1 on the validation set.

        Args:
            X_val: Validation features.
            y_val: Validation labels (0/1).

        Returns:
            Best threshold value.
        """
        probs = self.model.predict(X_val, verbose=0).flatten()
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            preds = (probs >= t).astype(int)
            tp = np.sum((preds == 1) & (y_val == 1))
            fp = np.sum((preds == 1) & (y_val == 0))
            fn = np.sum((preds == 0) & (y_val == 1))
            if (tp + fp) == 0 or (tp + fn) == 0:
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = best_t
        return best_t

    def predict(self, X):
        """Return binary predictions and raw probabilities.

        Args:
            X: Input features array.

        Returns:
            (predictions, probabilities) — both shape [n_samples]
        """
        probs = self.model.predict(X, verbose=0).flatten()
        preds = (probs >= self.threshold).astype(int)
        return preds, probs

    def evaluate(self, X, y):
        """Evaluate the detector and return a metrics dict.

        Args:
            X: Input features.
            y: True labels (0/1).

        Returns:
            Dict with precision, recall, f1, auc_pr keys.
        """
        preds, probs = self.predict(X)
        from siata_anomaly.metrics import precision_recall_f1
        return precision_recall_f1(y, preds, probs)
