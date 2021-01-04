from .utils import compare_metrics
from extra_keras_metrics import MatthewsCorrelationCoefficient
from sklearn.metrics import matthews_corrcoef as baseline
import numpy as np


def mcc_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_precision():
    compare_metrics(MatthewsCorrelationCoefficient(), mcc_score)
