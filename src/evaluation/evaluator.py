"""
Generic model evaluation utilities.

This module evaluates a trained binary classification model
in a consistent, model-agnostic way.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.evaluation.metrics import compute_classification_metrics


def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):

    # Probability for the positive class (class = 1)
    y_proba = model.predict_proba(X_test)[:, 1]

 
    # Apply decision threshold
    y_pred = (y_proba >= threshold).astype(int)


    # Compute scalar metrics
    metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)


    # Precision–Recall curve data
    precision, recall, pr_thresholds = precision_recall_curve(
        y_test, y_proba
    )

    # Collect results
    results = {
        "model_name": model_name,
        "threshold": threshold,
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresholds,
        },
        "pr_curve": {
            "precision": precision,
            "recall": recall,
            "thresholds": pr_thresholds,
        },
    }

    return results
