from .accuracy import Accuracy
from .balanced_accuracy import BalancedAccuracy
from .f1score import F1Score
from .matthews_correlation_coefficient import MatthewsCorrelationCoefficient
from .true_positives_ratio import TruePositivesRatio
from .false_positives_ratio import FalsePositivesRatio
from .true_negatives_ratio import TrueNegativesRatio
from .false_negatives_ratio import FalseNegativesRatio
from .recall import Recall
from .specificity import Specificity
from .precision import Precision
from .miss_rate import MissRate
from .fallout import FallOut
from .negative_predictive_value import NegativePredictiveValue
from .false_discovery_rate import FalseDiscoveryRate
from .false_omission_rate import FalseOmissionRate
from .prevalence_threshold import PrevalenceThreshold
from .threat_score import ThreatScore
from .fowlkes_mallows_index import FowlkesMallowsIndex
from .informedness import Informedness
from .markedness import Markedness
from .positive_likelihood_ratio import PositiveLikelihoodRatio
from .negative_likelihood_ratio import NegativeLikelihoodRatio
from .diagnostic_odds_ratio import DiagnosticOddsRatio

from tensorflow.keras.metrics import AUC

def get_binary_metrics():
    return [
            Accuracy(name="accuracy"),
            BalancedAccuracy(name="balanced_accuracy"),
            AUC(curve="roc", name="AUROC"),
            AUC(curve="pr", name="AUPRC"),
            F1Score(name="f1_score"),
            MatthewsCorrelationCoefficient(name="mcc"),
            TruePositivesRatio(name="tp/t"),
            FalsePositivesRatio(name="fp/t"),
            TrueNegativesRatio(name="tn/t"),
            FalseNegativesRatio(name="fn/t"),
            Recall(name="recall"),
            Specificity(name="specificity"),
            Precision(name="precision"),
            MissRate(name="miss_rate"),
            FallOut(name="fall_out"),
            NegativePredictiveValue(name="negative_predictive_value"),
            FalseDiscoveryRate(name="false_discovery_rate"),
            FalseOmissionRate(name="false_omission-rate"),
            PrevalenceThreshold(name="prevalence_threshold"),
            ThreatScore(name="threat_score"),
            FowlkesMallowsIndex(name="fowlkes_mallows_index"),
            Informedness(name="informedness"),
            Markedness(name="markedness"),
            PositiveLikelihoodRatio(name="LR_pos"),
            NegativeLikelihoodRatio(name="LR_neg"),
            DiagnosticOddsRatio(name="DOR")
    ]
