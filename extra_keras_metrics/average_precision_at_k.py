from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict

@parametrized_metric
def average_precision_at_k(k:int, *args:List, **kwargs:Dict):
    """Return a average_precision_at_k with parameter k.
        Integer, k for @k metric. This will calculate an average precision for range [1,k], as documented above.
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->float:
        """Return average_precision_at_k score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.average_precision_at_k(
            tf.to_int64(tf.math.round(labels)),
            tf.to_int64(tf.math.round(predictions)),
            k=k,
            *args,
            **kwargs
        )[1]
    return tmp