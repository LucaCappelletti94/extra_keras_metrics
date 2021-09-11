from tensorflow.keras import backend as K
from typing import Callable
import numpy as np


def run_metric(metric: Callable, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metric.reset_states()
    metric.update_state(y_true, y_pred)
    return metric.result()
