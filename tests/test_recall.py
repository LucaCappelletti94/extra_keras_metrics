import numpy as np
from extra_keras_metrics import Recall
from sklearn.metrics import recall_score as baseline

from .utils import compare_metrics


def recall_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))


def test_recall():
    compare_metrics(Recall(), recall_score)
