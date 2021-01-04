from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class FalsePositivesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.fp / (self.tp + self.fp + self.tn + self.fn + epsilon())
