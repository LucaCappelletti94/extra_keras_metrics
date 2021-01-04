from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class TrueNegativesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.tn / (self.tp + self.fp + self.tn + self.fn + epsilon())
