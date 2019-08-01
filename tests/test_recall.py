from .utils import compare_metrics
from extra_keras_metrics import recall
from sklearn.metrics import recall_score as baseline
import numpy as np

def recall_score(y_true, y_pred):
    return baseline(y_true, np.round(y_pred).astype(int))

def test_recall():
    compare_metrics(recall, recall_score)