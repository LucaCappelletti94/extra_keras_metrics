"""Module implementing the Diagnostic Odds Ratio metric."""
from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class DiagnosticOddsRatio(BinaryMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        fpr = self.fp / (self.fp + self.tn + epsilon())
        tnr = self.tn / (self.tn + self.fp + epsilon())
        fnr = self.fn / (self.fn + self.tp + epsilon())
        return (tpr + tnr) / (fpr + fnr + epsilon())
