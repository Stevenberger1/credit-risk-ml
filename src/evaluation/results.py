"""
Utilities for storing and comparing model evaluation results.
"""

import pandas as pd


def results_dict_to_row(results):
    """
    Convert a single model evaluation result dictionary into a flat table row.
    """

    row = {
        "model_name": results["model_name"],
        "threshold": results["threshold"],
        "accuracy": results["metrics"]["accuracy"],
        "precision": results["metrics"]["precision"],
        "recall": results["metrics"]["recall"],
        "roc_auc": results["metrics"]["roc_auc"],
        "pr_auc": results["metrics"]["pr_auc"],
    }

    return row


def append_results(results_df, results):
    """
    Append a new model evaluation result to the results DataFrame.
    """

    row = results_dict_to_row(results)

    if results_df is None:
        return pd.DataFrame([row])

    return pd.concat(
        [results_df, pd.DataFrame([row])],
        ignore_index=True,
    )


def sort_results(results_df, by="pr_auc", ascending=False):
    """
    Sort results by a chosen metric.
    """

    return results_df.sort_values(by=by, ascending=ascending)
