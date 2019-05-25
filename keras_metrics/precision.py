from .metric import metric
import tensorflow as tf

@metric
def precision(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return precision score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.precision(labels, predictions)[1]