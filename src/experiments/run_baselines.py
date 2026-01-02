"""
Run baseline models and collect evaluation results.
"""

from src.data import get_train_test_data
from src.preprocessing import build_preprocessor

from src.models.logistic_regression import build_logistic_regression_pipeline
# later:
# from src.models.random_forest import build_random_forest_pipeline
# from src.models.xgboost import build_xgboost_pipeline
# from src.models.lightgbm import build_lightgbm_pipeline

from src.evaluation.evaluator import evaluate_model
from src.evaluation.results import append_results, sort_results


def run_baselines():
    """
    Run baseline experiments for a list of models and return a results DataFrame.
    """

    # ------------------------------
    # 1. Load data
    # ------------------------------
    X_train, X_test, y_train, y_test = get_train_test_data()

    # ------------------------------
    # 2. Build preprocessing
    # ------------------------------
    preprocessor = build_preprocessor(X_train)

    # ------------------------------
    # 3. Define baseline models
    # ------------------------------
    baseline_models = [
        (
            "Logistic Regression (baseline)",
            build_logistic_regression_pipeline,
        ),
        # Add more models here later
    ]

    # ------------------------------
    # 4. Run experiments
    # ------------------------------
    results_df = None
    all_results = []

    for model_name, model_builder in baseline_models:
        print(f"Running baseline: {model_name}")

        # Build model pipeline
        model = model_builder(preprocessor)

        # Fit model
        model.fit(X_train, y_train)

        # Evaluate model
        results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            threshold=0.5,
        )

        all_results.append(results)
        results_df = append_results(results_df, results)

    # ------------------------------
    # 5. Sort results
    # ------------------------------
    results_df = sort_results(results_df, by="pr_auc", ascending=False)

    return results_df, all_results


if __name__ == "__main__":
    results = run_baselines()
    print("\nBaseline results:")
    print(results)
