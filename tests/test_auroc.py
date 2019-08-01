from .utils import compare_metrics
from extra_keras_metrics import auroc
from sklearn.metrics import roc_auc_score

def test_auroc():
    compare_metrics(auroc, roc_auc_score)