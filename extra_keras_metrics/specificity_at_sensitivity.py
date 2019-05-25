from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict

@parametrized_metric
def specificity_at_sensitivity(sensitivity:float, *args:List, **kwargs:Dict):
    """Return a specificity_at_sensitivity with parameter sensitivity.
        sensitivity:float, A scalar value in range [0, 1].
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->float:
        """Return specificity_at_sensitivity score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.specificity_at_sensitivity(
            labels,
            predictions,
            sensitivity=sensitivity,
            *args,
            **kwargs
        )[1]
    return tmp