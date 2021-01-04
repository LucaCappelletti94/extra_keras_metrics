import tensorflow as tf
from .binary_metric import BinaryMetric
from tensorflow.keras.backend import epsilon

class Informedness(BinaryMetric):
    def _custom_metric(self):
        tpr = self.tp / (self.tp + self.fn + epsilon())
        ppv = self.tp / (self.tp + self.fp + epsilon())
        return tf.math.sqrt(ppv * tpr)