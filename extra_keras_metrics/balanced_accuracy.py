"""Module implementing the Balanced Accuracy metric."""
from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class BalancedAccuracy(BinaryMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        tnr = self.tn / (self.tn + self.fp + epsilon())
        return (tpr + tnr) / 2
