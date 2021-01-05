import numpy as np
from extra_keras_metrics import Precision
from sklearn.metrics import precision_score as baseline

from .utils import compare_metrics


def precision_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_precision():
    compare_metrics(Precision(), precision_score)
