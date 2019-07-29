from .metric import metric
import tensorflow as tf
from typing import Dict, Tuple

@metric
def mean_absolute_error(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return mean_absolute_error score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.mean_absolute_error(labels, predictions)