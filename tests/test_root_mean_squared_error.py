from .utils import compare_metrics
from extra_keras_metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np

def baseline(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def test_root_mean_squared_error():
    compare_metrics(root_mean_squared_error, baseline)