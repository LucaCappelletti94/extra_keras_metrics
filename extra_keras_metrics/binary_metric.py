"""Module implementing the genetic binary metric class."""
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric, Sum
from tensorflow.python.keras.utils import metrics_utils


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
        self.tp_sum = Sum()
        self.fp_sum = Sum()
        self.tn_sum = Sum()
        self.fn_sum = Sum()

    @property
    def tp(self):
        return self.tp_sum.result()
    @property
    def fp(self):
        return self.fp_sum.result()
    @property
    def tn(self):
        return self.tn_sum.result()
    @property
    def fn(self):
        return self.fn_sum.result()

    def result(self):
        """Return the current state of the metric."""
        return self._custom_metric()

    def reset_state(self):
        """Reset the counters, this is called at the start of each epoch."""
        self.tp_sum.reset_state()
        self.fp_sum.reset_state()
        self.tn_sum.reset_state()
        self.fn_sum.reset_state()

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

        # Ensuring that the given tensors have compatible shapes
        [y_true, y_pred], sample_weight = (
            metrics_utils.ragged_assert_compatible_and_get_flat_values(
                [y_true, y_pred], sample_weight))

        y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        y_pred_neg = 1 - y_pred_pos

        y_pos = K.round(K.clip(y_true, 0, 1))
        y_neg = 1 - y_pos

        self.tp_sum.update_state(
            y_pos * y_pred_pos, 
            sample_weight=sample_weight,
        )
        self.fp_sum.update_state(
            y_neg * y_pred_pos, 
            sample_weight=sample_weight,
        )
        self.fn_sum.update_state(
            y_pos * y_pred_neg, 
            sample_weight=sample_weight,
        )
        self.tn_sum.update_state(
            y_neg * y_pred_neg, 
            sample_weight=sample_weight,
        )

        # The parent's method is an abstract method that is not implemented
        # in the parent class and thus it must not be called

    def _custom_metric(self):
        """Method that the subclasses should implement."""
        raise NotImplementedError(
            "The _custom_metric method must be implemented by subclasses."
        )
