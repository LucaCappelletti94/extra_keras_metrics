from .to_tensor import to_tensor
from keras import backend as K
from typing import Callable
import numpy as np


def run_metric(metric:Callable, y_true:np.ndarray, y_pred:np.ndarray)->float:
    return K.get_session().run(metric(*to_tensor(y_true, y_pred)))
