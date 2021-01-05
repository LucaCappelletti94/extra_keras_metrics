from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class ThreatScore(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fn + self.fp + epsilon())
