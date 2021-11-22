"""Module providing differential methods for metric lists."""
from typing import List, Union

from tensorflow.keras.metrics import AUC, Precision, Recall # pylint: disable=import-error,no-name-in-module
from tensorflow.keras.metrics import SparseCategoricalAccuracy # pylint: disable=import-error,no-name-in-module

from .balanced_accuracy import BalancedAccuracy
from .binary_metric import BinaryMetric
from .diagnostic_odds_ratio import DiagnosticOddsRatio
from .f1score import F1Score
from .fallout import FallOut
from .false_discovery_rate import FalseDiscoveryRate
from .false_negatives_ratio import FalseNegativesRatio
from .false_omission_rate import FalseOmissionRate
from .false_positives_ratio import FalsePositivesRatio
from .fowlkes_mallows_index import FowlkesMallowsIndex
from .informedness import Informedness
from .markedness import Markedness
from .matthews_correlation_coefficient import MatthewsCorrelationCoefficient
from .miss_rate import MissRate
from .negative_likelihood_ratio import NegativeLikelihoodRatio
from .negative_predictive_value import NegativePredictiveValue
from .positive_likelihood_ratio import PositiveLikelihoodRatio
from .prevalence_threshold import PrevalenceThreshold
from .specificity import Specificity
from .threat_score import ThreatScore
from .true_negatives_ratio import TrueNegativesRatio
from .true_positives_ratio import TruePositivesRatio


def get_sparse_multiclass_metrics() -> List[SparseCategoricalAccuracy]:
    """Return minimal list of multiclass metrics supporting sparse output.
    """
    return [
        SparseCategoricalAccuracy(name="sparse_categorical_accuracy")
    ]

def get_minimal_multiclass_metrics() -> List[Union[AUC, str, BinaryMetric]]:
    """Return minimal list of multiclass metrics.

    The set of metrics includes accuracy, AUROC and AUPRC.
    """
    return [
        "accuracy",
        Recall(name="recall"),
        Precision(name="precision"),
        AUC(curve="roc", name="AUROC"),
        AUC(curve="pr", name="AUPRC")
    ]


def get_standard_binary_metrics() -> List[Union[AUC, str, BinaryMetric]]:
    """Return standard list of binary metrics.

    The set of metrics includes accuracy, balanced accuracy, AUROC, AUPRC,
    F1 Score, Recall, Specificity, Precision, Miss rate,
    Fallout and Matthews Correlation Coefficient.
    """
    return [
        *get_minimal_multiclass_metrics(),
        F1Score(name="f1_score"),
        BalancedAccuracy(name="balanced_accuracy"),
        Specificity(name="specificity"),
        MissRate(name="miss_rate"),
        FallOut(name="fall_out"),
        MatthewsCorrelationCoefficient(name="mcc"),
    ]


def get_complete_binary_metrics() -> List[Union[AUC, str, BinaryMetric]]:
    """Return complete list of binary metrics.

    This method returns ALL the implemented metrics.

    The set of metrics includes accuracy, balanced accuracy, AUROC, AUPRC,
    F1 Score, Recall, Specificity, Precision, Miss rate,
    Fallout, Matthews Correlation Coefficient, rate of true positives over total,
    rate of false positive over total, rate of true negatives over total,
    rate of false negatives over total, negative predictive value,
    false discovery rate, false omission rate, prevalence threshold,
    threat score, Fowles-Mallows index, informedness, markedness,
    positive likelihood ratio, negative likelihood ratio and
    diagnostic odds ratio.
    """
    return [
        *get_standard_binary_metrics(),
        TruePositivesRatio(name="true_positives_over_total"),
        FalsePositivesRatio(name="false_positives_over_total"),
        TrueNegativesRatio(name="true_negatives_over_total"),
        FalseNegativesRatio(name="false_negatives_over_total"),
        NegativePredictiveValue(name="negative_predictive_value"),
        FalseDiscoveryRate(name="false_discovery_rate"),
        FalseOmissionRate(name="false_omission_rate"),
        PrevalenceThreshold(name="prevalence_threshold"),
        ThreatScore(name="threat_score"),
        FowlkesMallowsIndex(name="fowlkes_mallows_index"),
        Informedness(name="informedness"),
        Markedness(name="markedness"),
        PositiveLikelihoodRatio(name="positive_likelyhood_ratio"),
        NegativeLikelihoodRatio(name="negative_likelyhood_ratio"),
        DiagnosticOddsRatio(name="DOR")
    ]
