from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class FalseNegativesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.fn / (self.tp + self.fp + self.tn + self.fn + epsilon())
