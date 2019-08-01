from .utils import compare_metrics, true_positives as baseline
from extra_keras_metrics import true_positives

def test_true_positives():
    compare_metrics(true_positives, baseline, min_pearson_correlation=0.99999999999)