from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class Precision(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fp + epsilon())