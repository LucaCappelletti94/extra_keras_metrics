from .metric import metric
import tensorflow as tf
from typing import Tuple, Dict

@metric
def mean_squared_error(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return mean_squared_error score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.mean_squared_error(labels, predictions)