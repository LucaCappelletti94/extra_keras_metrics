from sklearn.metrics import confusion_matrix
import numpy as np


def _ravel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    return confusion_matrix(y_true, np.round(y_pred).astype(int)).ravel()


def true_negatives(y_true: np.ndarray, y_pred: np.ndarray):
    return _ravel_confusion_matrix(y_true, y_pred)[0]


def false_positives(y_true: np.ndarray, y_pred: np.ndarray):
    return _ravel_confusion_matrix(y_true, y_pred)[1]


def false_negatives(y_true: np.ndarray, y_pred: np.ndarray):
    return _ravel_confusion_matrix(y_true, y_pred)[2]


def true_positives(y_true: np.ndarray, y_pred: np.ndarray):
    return _ravel_confusion_matrix(y_true, y_pred)[3]
