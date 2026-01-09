"""
Run baseline models and collect evaluation results.
"""

from src.data import get_train_test_data
from src.preprocessing import build_preprocessor

from src.models.logistic_regression import build_logistic_regression_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost import build_xgboost_pipeline
from src.models.lightgbm import build_lightgbm_pipeline

from src.evaluation.evaluator import evaluate_model
from src.evaluation.results import append_results, sort_results


# ============================================================
# Baseline model registry (CONFIGURATION ONLY)
# ============================================================

BASELINE_MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression (baseline)",
        "builder": build_logistic_regression_pipeline,
    },
    "random_forest": {
        "name": "Random Forest (baseline)",
        "builder": build_random_forest_pipeline,
    },
    "xgboost": {
        "name": "XGBoost (baseline)",
        "builder": build_xgboost_pipeline,
    },
    "lightgbm": {
        "name": "LightGBM (baseline)",
        "builder": build_lightgbm_pipeline,
    },
}



# ============================================================
# Experiment runner
# ============================================================

def run_baselines(models_to_run=None, threshold=0.5):
    """
    Run baseline experiments.

    Parameters
    ----------
    models_to_run : list[str] or None
        List of model keys to run.
        If None, all baseline models are run.
    threshold : float
        Decision threshold for classification.

    Returns
    -------
    results_df : pd.DataFrame
        Summary metrics table.
    all_results : list[dict]
        Full evaluation artifacts for each model.
    """

    # ------------------------------
    # 1. Select models
    # ------------------------------
    if models_to_run is None:
        selected_models = BASELINE_MODELS
    else:
        selected_models = {
            k: BASELINE_MODELS[k]
            for k in models_to_run
        }

    # ------------------------------
    # 2. Load data
    # ------------------------------
    X_train, X_test, y_train, y_test = get_train_test_data()

    # ------------------------------
    # 3. Build preprocessing
    # ------------------------------
    preprocessor = build_preprocessor(X_train)

    # ------------------------------
    # 4. Run experiments
    # ------------------------------
    results_df = None
    all_results = []

    for model_key, model_cfg in selected_models.items():
        model_name = model_cfg["name"]
        model_builder = model_cfg["builder"]

        print(f"Running baseline: {model_name}")

        model = model_builder(preprocessor)
        model.fit(X_train, y_train)

        results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            threshold=threshold,
        )

        all_results.append(results)
        results_df = append_results(results_df, results)

    # ------------------------------
    # 5. Sort results
    # ------------------------------
    results_df = sort_results(results_df, by="pr_auc", ascending=False)

    return results_df, all_results


# ============================================================
# Optional CLI entry point
# ============================================================

if __name__ == "__main__":
    results_df, _ = run_baselines(
        )
    print("\nBaseline results:")
    print(results_df)

