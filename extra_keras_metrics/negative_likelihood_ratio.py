from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class NegativeLikelihoodRatio(BinaryMetric):
    def _custom_metric(self):
        tnr = self.tn / (self.tn + self.fp + epsilon())
        fnr = self.fn / (self.fn + self.tp + epsilon())
        return fnr / (tnr + epsilon())