from typing import Callable
import numpy as np
from .run_metric import run_metric


def compare_metrics(
    tensorflow_metric: Callable,
    baseline_metric: Callable,
    tests: int = 10,
    min_pearson_correlation: float = 0.99,
):
    """Run comparison test for the two given metrics.
    tensorflow_metric:Callable, metric to be tested.
    baseline_metric:Callable, metric to use as baseline.
    tests:int=20, number of test runs.
    min_pearson_correlation:float=0.01, minimum pearson correlation.
    """
    size = (tests, 100, 1)
    y_true = np.random.randint(2, size=size) / 1.0
    N = 100000
    y_pred = np.random.randint(N, size=size) / N

    scores = np.array(
        [
            (
                run_metric(tensorflow_metric, y_true[i], y_pred[i]),
                baseline_metric(y_true[i], y_pred[i]),
            )
            for i in range(tests)
        ]
    ).T
    pearson = np.corrcoef(scores).ravel()[1]

    assert pearson >= min_pearson_correlation
