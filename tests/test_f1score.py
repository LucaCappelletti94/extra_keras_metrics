from .utils import compare_metrics
from extra_keras_metrics import F1Score
from sklearn.metrics import f1_score as baseline
import numpy as np


def f1score_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_precision():
    compare_metrics(F1Score(), f1score_score)
