from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class Specificity(BinaryMetric):
    def _custom_metric(self):
        return self.tn / (self.tn + self.fp + epsilon())
