from .metric import metric
import tensorflow as tf

@metric
def true_positives(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return true_positives score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.true_positives(labels, predictions)[1]