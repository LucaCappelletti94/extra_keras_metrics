from .utils import compare_metrics
from extra_keras_metrics import Accuracy
from sklearn.metrics import accuracy_score as baseline
import numpy as np


def acc_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_precision():
    compare_metrics(Accuracy(), acc_score)
