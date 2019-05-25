from .metric import metric
import tensorflow as tf

@metric
def root_mean_squared_error(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return root_mean_squared_error score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.root_mean_squared_error(labels, predictions)[1]