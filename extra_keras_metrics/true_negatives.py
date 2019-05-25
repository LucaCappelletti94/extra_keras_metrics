from .metric import metric
import tensorflow as tf

@metric
def true_negatives(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return true_negatives score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.true_negatives(labels, predictions)[1]