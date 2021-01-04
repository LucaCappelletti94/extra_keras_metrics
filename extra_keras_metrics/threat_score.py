from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class ThreatScore(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fn + self.fp + epsilon())