from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict, Callable, Tuple


@parametrized_metric
def mean_iou(num_classes: int, *args: List) -> Callable[[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, Dict]]:
    """Return a mean_iou with parameter k.
        num_classes: The possible number of labels the prediction task can have.
    """
    @metric
    def tmp(labels: tf.Tensor, predictions: tf.Tensor) -> Tuple[tf.Tensor, Dict]:
        """Return mean_iou score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.mean_iou(
            tf.to_int64(tf.math.round(labels)),
            tf.to_int64(tf.math.round(predictions)),
            num_classes=num_classes,
            *args
        )
    return tmp
