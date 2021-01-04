from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class Accuracy(BinaryMetric):
    def _custom_metric(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + epsilon())
        