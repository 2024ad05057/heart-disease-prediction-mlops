"""
evaluate.py

Utility functions for evaluating machine learning models.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def evaluate_model(y_true, y_pred, y_prob):
    """
    Evaluate classification model.
    """

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob)
    }


def print_report(y_true, y_pred):
    """
    Print classification report.
    """

    print(classification_report(y_true, y_pred))