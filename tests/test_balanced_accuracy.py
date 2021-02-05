import numpy as np
from extra_keras_metrics import BalancedAccuracy
from sklearn.metrics import balanced_accuracy_score as baseline

from .utils import compare_metrics


def ba_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_precision():
    compare_metrics(BalancedAccuracy(), ba_score)
