from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class FalsePositivesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.fp / (self.tp + self.fp + self.tn + self.fn + epsilon())
