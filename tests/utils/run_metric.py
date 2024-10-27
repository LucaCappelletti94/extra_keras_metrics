from typing import Callable
from tensorflow.keras import backend as K
import numpy as np


def run_metric(metric: Callable, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Run a metric on the given data."""
    metric.reset_state()
    metric.update_state(y_true, y_pred)
    return metric.result()
