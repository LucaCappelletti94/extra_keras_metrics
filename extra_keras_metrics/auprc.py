from .metric import metric
from typing import Tuple, Dict
import tensorflow as tf


@metric
def auprc(labels: tf.Tensor, predictions: tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return auprc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels, predictions, curve="PR", summation_method="careful_interpolation")