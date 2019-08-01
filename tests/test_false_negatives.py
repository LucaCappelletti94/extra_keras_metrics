from .utils import compare_metrics, false_negatives as baseline
from extra_keras_metrics import false_negatives

def test_false_negatives():
    compare_metrics(false_negatives, baseline, min_pearson_correlation=0.99999999999)