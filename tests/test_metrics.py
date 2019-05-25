from extra_keras_metrics import metrics, parametric_metrics, non_parametric_metrics
from .utils import train_with_metric
import numpy as np
import tensorflow as tf

params = {
    "average_precision_at_k":{"args":[1,], "kwargs":{}},
    "precision_at_k":{"args":[], "kwargs":{"k":1}},
    "precision_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "recall_at_k":{"args":[1,], "kwargs":{}},
    "recall_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "sensitivity_at_specificity":{"args":[0.9,], "kwargs":{}},
    "specificity_at_sensitivity":{"args":[0.9,], "kwargs":{}},
    "false_negatives_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "true_negatives_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "false_positives_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "true_positives_at_thresholds":{"args":[[0.9],], "kwargs":{}},
    "mean_cosine_distance":{"args":[1,], "kwargs":{}},
    "mean_iou":{"args":[2,], "kwargs":{}},
    "mean_per_class_accuracy":{"args":[2,], "kwargs":{}},
    "mean_relative_error":{"args":[tf.constant(np.ones((10,1)), dtype=np.float32),], "kwargs":{}},
}

def test_metrics():
    global params
    assert all([
        m.__name__ in params for m in parametric_metrics
    ])
    assert all([
        m.__name__ not in params for m in non_parametric_metrics
    ])
    for metric in metrics:
        name = metric.__name__
        print("Testing {metric}".format(metric=name))
        train_with_metric(
            metric(
                *params[name]["args"],
                **params[name]["kwargs"]
            ) if name in params else name
        )