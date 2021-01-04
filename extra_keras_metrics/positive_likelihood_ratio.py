from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class PositiveLikelihoodRatio(BinaryMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        fpr = self.fp / (self.fp + self.tn + epsilon())
        return tpr / (fpr + epsilon())