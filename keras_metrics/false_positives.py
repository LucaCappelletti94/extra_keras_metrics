from .metric import metric
import tensorflow as tf

@metric
def false_positives(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return false_positives score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.false_positives(labels, predictions)[1]