import keras
from .average_precision_at_k import average_precision_at_k
from .auprc import auprc
from .auroc import auroc
from .false_negatives import false_negatives
from .false_positives import false_positives
from .mean_absolute_error import mean_absolute_error
from .mean_squared_error import mean_squared_error
from .precision import precision
from .precision_at_k import precision_at_k
from .recall import recall
from .recall_at_k import recall_at_k
from .root_mean_squared_error import root_mean_squared_error
from .sensitivity_at_specificity import sensitivity_at_specificity
from .specificity_at_sensitivity import specificity_at_sensitivity
from .true_negatives import true_negatives
from .true_positives import true_positives

non_parametric_metrics = [
    auprc,
    auroc,
    false_negatives,
    false_positives,
    mean_absolute_error,
    mean_squared_error,
    precision,
    recall,
    root_mean_squared_error,
    true_negatives,
    true_positives
]

non_parametric_metrics_names = {m.__name__: m for m in non_parametric_metrics}

old_get = keras.metrics.get


def get(identifier):
    global non_parametric_metrics_names
    if identifier in non_parametric_metrics_names:
        return non_parametric_metrics_names[identifier]
    return old_get(identifier)


keras.metrics.get = get


__all__ = list(non_parametric_metrics_names.keys()) + [
    "average_precision_at_k",
    "precision_at_k",
    "recall_at_k",
    "sensitivity_at_specificity",
    "specificity_at_sensitivity"
]
