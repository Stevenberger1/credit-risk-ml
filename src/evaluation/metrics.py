"""
Evaluation metrics for classification models.

This module defines a consistent set of metrics that are used
to evaluate all models in the project.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


def compute_classification_metrics(y_true, y_pred, y_proba):
    """
    Compute core classification metrics.

    y_proba = model.predict_proba(X_test)[:, 1] - How confident is the model that this sample belongs to the positive class?

   """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }

    return metrics
