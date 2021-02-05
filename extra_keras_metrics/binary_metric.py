"""Module implementing the genetic binary metric class."""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric


class BinaryMetric(Metric):
    """**BINARY** Metric which is based on the true positives, false positives, 
    true negatives, false negatives.

    This class is abstract and its childs should override the `_custom_metric`
    method.

    The method can use the variables:
        self.tp -> true positvies
        self.fp -> false positvies
        self.tn -> true negatives
        self.fn -> false negatives

    And the result should return the compute value.

    Example:

    class Recall(BinaryMetric):
        def _custom_metric(self):
            return self.tp / (self.tp + self.fn + epsilon())
    """

    def __init__(self, name: str = None, **kwargs):
        """Initialize the Binary metric. The kwargs will be passed to the father
            class tensorflow.keras.metrics.Metric.

        Parameters
        ---------
        name: str,
            The name of the metric.
        """
        super(BinaryMetric, self).__init__(
            name=name or self.__class__.__name__,
            **kwargs
        )
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self._result = self.add_weight(name=name, initializer='zeros')

    def result(self):
        """Return the current state of the metric."""
        return self._result

    def reset_states(self):
        """Reset the counters, this is called at the start of each epoch."""
        self.tp.assign(0)
        self.fp.assign(0)
        self.tn.assign(0)
        self.fn.assign(0)
        self._result.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Given the predictions of the new batch, update the counters of the 
        metric. This method is not supposed to return anything but it must
        only update attributes.

        Parameters
        ---------
        y_true: tf.tensor,
            The ground truth.
        y_pred: tf.tensor,
            The predictions of the model.
        sample_weight: float,
            This parameter is standard for the metrics but it's not used in
            this class. (IGNORED)
        """
        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        self.tp.assign_add(K.sum(y_pos * y_pred_pos))
        self.fp.assign_add(K.sum(y_neg * y_pred_pos))
        self.fn.assign_add(K.sum(y_pos * y_pred_neg))
        self.tn.assign_add(K.sum(y_neg * y_pred_neg))

        self._result.assign(self._custom_metric())

    def _custom_metric(self):
        """Method that the subclasses should implement."""
        raise NotImplementedError(
            "The _custom_metric method must be implemented by subclasses."
        )
