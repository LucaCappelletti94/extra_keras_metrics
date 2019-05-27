from .metric import metric
import tensorflow as tf

@metric
def auprc(labels:tf.Tensor, predictions:tf.Tensor)->float:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels, predictions, curve="PR", summation_method="careful_interpolation")[1]