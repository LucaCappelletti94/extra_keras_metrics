from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict, Callable, Tuple


@parametrized_metric
def mean_relative_error(normalizer: tf.Tensor, *args: List, **kwargs: Dict)->Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    """Return a mean_relative_error with parameter normalizer.
        normalizer:Tensor: The possible number of labels the prediction task can have. This value must be provided, since two variables with shape = [normalizer] will be allocated.
    """
    @metric
    def tmp(labels: tf.Tensor, predictions: tf.Tensor)->Tuple[tf.Tensor, Dict]:
        """Return mean_relative_error score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.mean_relative_error(labels, predictions, normalizer=normalizer, *args, **kwargs)
    return tmp
