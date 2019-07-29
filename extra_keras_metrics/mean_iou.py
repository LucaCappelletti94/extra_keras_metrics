from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict, Tuple, Callable

@parametrized_metric
def mean_iou(num_classes:int, *args:List, **kwargs:Dict)->Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    """Return a mean_iou with parameter num_classes.
        num_classes:int: The possible number of labels the prediction task can have. This value must be provided, since a confusion matrix of dimension = [num_classes, num_classes] will be allocated.
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->Tuple[tf.Tensor, Dict]:
        """Return mean_iou score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.mean_iou(labels, predictions, num_classes=num_classes, *args, **kwargs)
    return tmp