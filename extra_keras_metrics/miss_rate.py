from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class MissRate(BinaryMetric):
    def _custom_metric(self):
        return self.fn / (self.fn + self.tp + epsilon())
