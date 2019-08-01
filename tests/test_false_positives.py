from .utils import compare_metrics, false_positives as baseline
from extra_keras_metrics import false_positives

def test_false_positives():
    compare_metrics(false_positives, baseline, min_pearson_correlation=1)