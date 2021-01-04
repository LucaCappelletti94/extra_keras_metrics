from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class FallOut(BinaryMetric):
    def _custom_metric(self):
        return self.fp / (self.fp + self.tn + epsilon())