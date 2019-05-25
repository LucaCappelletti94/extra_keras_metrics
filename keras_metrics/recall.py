from .metric import metric
import tensorflow as tf

@metric
def recall(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return recall score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.recall(labels, predictions)[1]