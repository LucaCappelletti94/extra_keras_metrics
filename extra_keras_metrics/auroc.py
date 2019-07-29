from .metric import metric
import tensorflow as tf
from typing import Tuple, Dict


@metric
def auroc(labels: tf.Tensor, predictions: tf.Tensor)->Tuple[tf.Tensor, Dict]:
    """Return auroc score for given epoch results.
        labels:tf.Tensor, the expected output values.
        predictions:tf.Tensor, the predicted output values.
    """
    return tf.metrics.auc(labels, predictions, curve="ROC", summation_method="careful_interpolation")
