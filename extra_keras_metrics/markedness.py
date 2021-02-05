from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class Markedness(BinaryMetric):
    def _custom_metric(self):
        ppv = self.tp / (self.tp + self.fp + epsilon())
        npv = self.tn / (self.tn + self.fn + epsilon())
        return ppv + npv - 1
