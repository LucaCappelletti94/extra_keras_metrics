from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class PositiveLikelihoodRatio(BinaryMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        fpr = self.fp / (self.fp + self.tn + epsilon())
        return tpr / (fpr + epsilon())
