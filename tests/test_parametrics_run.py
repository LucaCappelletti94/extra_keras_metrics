from .utils import run_metric
from extra_keras_metrics import average_precision_at_k, precision_at_k, recall_at_k, sensitivity_at_specificity, specificity_at_sensitivity, mean_iou
import numpy as np

def test_parametrics_run():
    size = (100, 1)
    y_true = np.random.randint(2, size=size)/1.0
    N = 100000
    y_pred = np.random.randint(N, size=size)/N
    run_metric(average_precision_at_k(k=1), y_true, y_pred)
    run_metric(precision_at_k(k=1), y_true, y_pred)
    run_metric(recall_at_k(k=1), y_true, y_pred)
    run_metric(sensitivity_at_specificity(specificity=0.3), y_true, y_pred)
    run_metric(specificity_at_sensitivity(sensitivity=0.3), y_true, y_pred)
    run_metric(mean_iou(num_classes=2), y_true, y_pred)