from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class FalseNegativesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.fn / (self.tp + self.fp + self.tn + self.fn + epsilon())
