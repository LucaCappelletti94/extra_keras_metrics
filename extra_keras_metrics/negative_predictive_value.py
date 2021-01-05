from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class NegativePredictiveValue(BinaryMetric):
    def _custom_metric(self):
        return self.tn / (self.fn + self.tn + epsilon())
