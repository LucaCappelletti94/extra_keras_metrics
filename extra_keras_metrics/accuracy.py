from .metric import metric
import tensorflow as tf

@metric
def accuracy(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return accuracy score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.accuracy(labels, predictions)[1]