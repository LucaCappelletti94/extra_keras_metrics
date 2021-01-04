from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class FalseDiscoveryRate(BinaryMetric):
    def _custom_metric(self):
        return self.fp / (self.fp + self.tp + epsilon())