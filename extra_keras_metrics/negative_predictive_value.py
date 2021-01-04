from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class NegativePredictiveValue(BinaryMetric):
    def _custom_metric(self):
        return self.tn / (self.fn + self.tn + epsilon())