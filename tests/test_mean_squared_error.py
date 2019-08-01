from .utils import compare_metrics
from extra_keras_metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as baseline

def test_mean_squared_error():
    compare_metrics(mean_squared_error, baseline)