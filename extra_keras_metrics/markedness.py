from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class Markedness(BinaryMetric):
    def _custom_metric(self):
        ppv = self.tp / (self.tp + self.fp + epsilon())
        npv = self.tn / (self.tn + self.fn + epsilon())
        return ppv + npv - 1