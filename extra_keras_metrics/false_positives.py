from .metric import metric
import tensorflow as tf
from typing import Tuple, Dict

@metric
def false_positives(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return false_positives score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.false_positives(labels, tf.to_int64(tf.math.round(predictions)))