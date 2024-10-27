"""Module providing differential methods for metric lists."""

from typing import List, Union

from tensorflow.keras.metrics import (  # pylint: disable=import-error,no-name-in-module
    AUC,
    Precision,
    Recall,
)
from tensorflow.keras.metrics import (  # pylint: disable=import-error,no-name-in-module
    SparseCategoricalAccuracy,
)

from extra_keras_metrics.balanced_accuracy import BalancedAccuracy
from extra_keras_metrics.binary_metric import BinaryMetric
from extra_keras_metrics.diagnostic_odds_ratio import DiagnosticOddsRatio
from extra_keras_metrics.f1score import F1Score
from extra_keras_metrics.fallout import FallOut
from extra_keras_metrics.false_discovery_rate import FalseDiscoveryRate
from extra_keras_metrics.false_negatives_ratio import FalseNegativesRatio
from extra_keras_metrics.false_omission_rate import FalseOmissionRate
from extra_keras_metrics.false_positives_ratio import FalsePositivesRatio
from extra_keras_metrics.fowlkes_mallows_index import FowlkesMallowsIndex
from extra_keras_metrics.informedness import Informedness
from extra_keras_metrics.markedness import Markedness
from extra_keras_metrics.matthews_correlation_coefficient import (
    MatthewsCorrelationCoefficient,
)
from extra_keras_metrics.miss_rate import MissRate
from extra_keras_metrics.negative_likelihood_ratio import NegativeLikelihoodRatio
from extra_keras_metrics.negative_predictive_value import NegativePredictiveValue
from extra_keras_metrics.positive_likelihood_ratio import PositiveLikelihoodRatio
from extra_keras_metrics.prevalence_threshold import PrevalenceThreshold
from extra_keras_metrics.specificity import Specificity
from extra_keras_metrics.threat_score import ThreatScore
from extra_keras_metrics.true_negatives_ratio import TrueNegativesRatio
from extra_keras_metrics.true_positives_ratio import TruePositivesRatio


def get_sparse_multiclass_metrics() -> List[SparseCategoricalAccuracy]:
    """Return minimal list of multiclass metrics supporting sparse output."""
    return [SparseCategoricalAccuracy(name="sparse_categorical_accuracy")]


def get_minimal_multiclass_metrics() -> List[Union[AUC, str, BinaryMetric]]:
    """Return minimal list of multiclass metrics.

    The set of metrics includes accuracy, AUROC and AUPRC.
    """
    return [
        "accuracy",
        Recall(name="recall"),
        Precision(name="precision"),
        AUC(curve="roc", name="AUROC"),
        AUC(curve="pr", name="AUPRC"),
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
        DiagnosticOddsRatio(name="DOR"),
    ]
