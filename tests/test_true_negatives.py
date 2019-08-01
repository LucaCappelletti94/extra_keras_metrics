from .utils import compare_metrics, true_negatives as baseline
from extra_keras_metrics import true_negatives

def test_true_negatives():
    compare_metrics(true_negatives, baseline, min_pearson_correlation=1)