from typing import Callable, Dict
import numpy as np
from .to_tensor import to_tensor
from keras import backend as K


def compare_metrics(tensorflow_metric:Callable, baseline_metric:Callable, baseline_metric_kwargs:Dict=None, tests:int=10, min_pearson_correlation:float=0.99):
    """Run comparison test for the two given metrics.
        tensorflow_metric:Callable, metric to be tested.
        baseline_metric:Callable, metric to use as baseline.
        baseline_metric_kwargs:Dict=None, arguments to pass to the baseline metric.
        tests:int=20, number of test runs.
        min_pearson_correlation:float=0.01, minimum pearson correlation.
    """
    size = (tests, 100, 1)
    y_true = np.random.randint(2, size=size)/1.0
    N = 100000
    y_pred = np.random.randint(N, size=size)/N

    if baseline_metric_kwargs is None:
        baseline_metric_kwargs = {}

    scores = np.array([
        (
            K.get_session().run(tensorflow_metric(*to_tensor(y_true[i], y_pred[i]))),
            baseline_metric(y_true[i], y_pred[i], **baseline_metric_kwargs)
        ) for i in range(tests)
    ]).T
    pearson = np.corrcoef(scores).ravel()[1]

    try:
        assert pearson >= min_pearson_correlation
    except AssertionError as e:
        print("Test for {metric} failed!".format(metric=tensorflow_metric.__name__))
        print("Score: {corr}".format(corr=pearson))
        print(scores)
        raise e