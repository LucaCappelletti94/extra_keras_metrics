import tensorflow as tf
from tensorflow.keras.backend import epsilon

from .binary_metric import BinaryMetric


class MatthewsCorrelationCoefficient(BinaryMetric):
    def _custom_metric(self):
        numerator = (self.tp * self.tn - self.fp * self.fn)
        denominator = tf.math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))
        return numerator / (denominator + epsilon())
