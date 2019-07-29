from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict, Callable, Tuple

@parametrized_metric
def recall_at_thresholds(thresholds:List[float], *args: List, **kwargs: Dict)->Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    """Return a recall_at_thresholds with parameter thresholds.
        thresholds:List[float], A python list or tuple of float thresholds in [0, 1].
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
        """Return recall_at_thresholds score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.recall_at_thresholds(labels, predictions, thresholds=thresholds, *args, **kwargs)
    return tmp