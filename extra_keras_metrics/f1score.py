"""Module implementing the F1 Score metric."""
from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class F1Score(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + 0.5 * (self.fp + self.fn) + epsilon())
