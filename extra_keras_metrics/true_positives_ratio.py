from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class TruePositivesRatio(BinaryMetric):
    def _custom_metric(self):
        return self.tp / (self.tp + self.fp + self.tn + self.fn + epsilon())
