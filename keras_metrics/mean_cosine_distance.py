from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict

@parametrized_metric
def mean_cosine_distance(dim:int, *args:List, **kwargs:Dict):
    """Return a mean_cosine_distance with parameter dim.
        dim:int, The dimension along which the cosine distance is computed.
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->float:
        """Return mean_cosine_distance score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.mean_cosine_distance(labels, predictions, dim=dim, *args, **kwargs)[1]
    return tmp