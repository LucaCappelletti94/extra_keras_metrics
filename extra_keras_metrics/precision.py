from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class Precision(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fp + epsilon())
