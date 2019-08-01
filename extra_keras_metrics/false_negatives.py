from .metric import metric
import tensorflow as tf
from typing import Dict, Tuple

@metric
def false_negatives(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return false_negatives score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.false_negatives(labels, tf.to_int64(tf.math.round(predictions)))