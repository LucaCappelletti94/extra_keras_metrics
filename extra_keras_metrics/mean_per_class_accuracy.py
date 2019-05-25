from .metric import metric
from .parametrized_metric import parametrized_metric
import tensorflow as tf
from typing import List, Dict

@parametrized_metric
def mean_per_class_accuracy(num_classes:int, *args:List, **kwargs:Dict):
    """Return a mean_per_class_accuracy with parameter num_classes.
        num_classes:int: The possible number of labels the prediction task can have. This value must be provided, since two variables with shape = [num_classes] will be allocated.
    """
    @metric
    def tmp(labels:tf.Tensor, predictions:tf.Tensor)->float:
        """Return mean_per_class_accuracy score for given epoch results.
            labels:tf.Tensor, the expected output values.
            predictions:tf.Tensor, the predicted output values.
        """
        return tf.metrics.mean_per_class_accuracy(labels, predictions, num_classes=num_classes, *args, **kwargs)[1]
    return tmp