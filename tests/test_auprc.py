from .utils import compare_metrics
from extra_keras_metrics import auprc
from sklearn.metrics import average_precision_score

def test_auprc():
    compare_metrics(auprc, average_precision_score)