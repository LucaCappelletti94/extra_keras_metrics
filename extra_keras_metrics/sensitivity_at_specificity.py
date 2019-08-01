from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict, Callable, Tuple

@parametrized_metric
def sensitivity_at_specificity(specificity:float, *args: List)->Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    """Return a sensitivity_at_specificity with parameter specificity.
        specificity:float, A scalar value in range [0, 1].
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
        """Return sensitivity_at_specificity score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.sensitivity_at_specificity(
            labels,
            predictions,
            specificity=specificity,
            *args
        )
    return tmp