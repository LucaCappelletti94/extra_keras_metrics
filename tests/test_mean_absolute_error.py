from .utils import compare_metrics
from extra_keras_metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error as baseline

def test_mean_absolute_error():
    compare_metrics(mean_absolute_error, baseline)