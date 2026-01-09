"""
Run hyperparameter-tuned models (CV on train) and evaluate once on test.

Key idea:
- Hyperparameter search happens ONLY on the training split using CV.
- The test split is touched only once at the end for final evaluation.
"""

# 1) Standard library imports (control reproducibility + types)
from __future__ import annotations # Allows modern type hints (list[str], dict[str, ...]) to work more smoothly across Python versions.

# 2) Third-party imports (data + model selection)
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
#RandomizedSearchCV: samples hyperparameter combos and cross-validates them.
#StratifiedKFold: creates CV folds preserving class proportions.

# 3) Project imports (your existing modules)
from src.config import ARTIFACT_DIR_Tuned
from src.data import get_train_test_data
from src.preprocessing import build_preprocessor

from src.models.logistic_regression import build_logistic_regression_pipeline
from src.models.random_forest import build_random_forest_pipeline
from src.models.xgboost import build_xgboost_pipeline
from src.models.lightgbm import build_lightgbm_pipeline

from src.evaluation.evaluator import evaluate_model
from src.evaluation.results import append_results, sort_results


from src.evaluation.persistence import (
    save_model_artifact,
    save_or_update_summary,
)

from src.config import ARTIFACT_DIR_Tuned


# ============================================================
# 4) Experiment configuration (outside the function)
#    This is the "what to run" section, not the "how to run".
# ============================================================

# 4.1) Cross-validation setup used during tuning (train only)
CV_FOLDS = 5  # Why: enough stability without being too slow
CV_RANDOM_STATE = 42  # Why: reproducibility

# 4.2) Optimization target for tuning
# For imbalanced problems, PR-AUC (average precision) is often more meaningful than accuracy.
TUNING_SCORING = "average_precision"

# 4.3) Random search budget (you can raise later)
DEFAULT_N_ITER = 25  # Why: small-ish baseline to keep runtime manageable, default number of random hyperparameter samples.

# 4.4) Tuned model registry:
# - key: stable ID you can select from CLI/notebooks
# - name: pretty display name
# - builder: builds a full sklearn Pipeline(preprocessor -> classifier)
# - params: search space (note: keys use "classifier__" because inside Pipeline)
TUNED_MODELS = {
    "logistic_regression": {
        "name": "Logistic Regression (tuned)",
        "builder": build_logistic_regression_pipeline,
        "n_iter": 20,
        "params": {
            # Regularization strength (C smaller => stronger regularization)
            "classifier__C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], # tunes regularization strength.
            # Solver must support the penalty you choose; liblinear supports l1/l2 in binary
            "classifier__penalty": ["l1", "l2"],
            "classifier__solver": ["liblinear"], # stable choice for small/medium datasets
        },
    },
    "random_forest": {
        "name": "Random Forest (tuned)",
        "builder": build_random_forest_pipeline,
        "n_iter": DEFAULT_N_ITER,
        "params": {
            "classifier__n_estimators": [200, 400, 600, 800],
            "classifier__max_depth": [None, 3, 5, 8, 12, 16],
            "classifier__min_samples_split": [2, 5, 10, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8],
            "classifier__max_features": ["sqrt", "log2", None],
        },
    },
    "xgboost": {
        "name": "XGBoost (tuned)",
        "builder": build_xgboost_pipeline,
        "n_iter": DEFAULT_N_ITER,
        "params": {
            "classifier__n_estimators": [200, 400, 600, 800],
            "classifier__max_depth": [2, 3, 4, 5, 6, 8],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
            "classifier__reg_lambda": [0.0, 0.5, 1.0, 2.0, 5.0],
        },
    },
    "lightgbm": {
        "name": "LightGBM (tuned)",
        "builder": build_lightgbm_pipeline,
        "n_iter": DEFAULT_N_ITER,
        "params": {
            "classifier__n_estimators": [200, 400, 600, 800],
            "classifier__learning_rate": [0.02, 0.05, 0.1, 0.2],
            "classifier__num_leaves": [15, 31, 63, 127],
            "classifier__max_depth": [-1, 3, 5, 8, 12],
            "classifier__subsample": [0.6, 0.8, 1.0],
            "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        },
    },
}


# ============================================================
# 5) Runner function (execution logic only)
# ============================================================

def run_tuned_models(models_to_run: list[str] | None = None, threshold: float = 0.5):
    """
    Tune models with CV on training set, then evaluate best models on the test set.

    Parameters
    ----------
    models_to_run : list[str] or None
        If None => run all models in TUNED_MODELS.
        Else => run only these keys, e.g. ["random_forest", "xgboost"].
    threshold : float
        Classification threshold used in final evaluation (not during tuning).

    Returns
    -------
    results_df : pd.DataFrame
        Summary table of final test metrics for each tuned model.
    all_results : list[dict]
        Rich artifacts (curves, confusion matrices) for visualization.
    best_params_df : pd.DataFrame
        Best hyperparameters found for each tuned model.
    """

    # 5.1) Choose which tuned models to run
    if models_to_run is None:
        selected = TUNED_MODELS
    else:
        selected = {k: TUNED_MODELS[k] for k in models_to_run}

    # 5.2) Load your fixed train/test split (test is held out)
    X_train, X_test, y_train, y_test = get_train_test_data()

    # 5.3) Build preprocessing using training data only (prevents leakage)
    preprocessor = build_preprocessor(X_train)

    # 5.4) Define CV strategy (stratified keeps class ratio similar per fold)
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    # 5.5) Collect outputs
    results_df = None
    all_results = []
    best_params_rows = []

    # 5.6) Loop over models and tune each one
    for model_key, cfg in selected.items():
        model_name = cfg["name"]
        builder = cfg["builder"]
        params = cfg["params"]
        n_iter = cfg.get("n_iter", DEFAULT_N_ITER)

        print(f"\nTuning: {model_name}")

        # 5.6.1) Build the baseline pipeline (preprocessor + classifier)
        pipeline = builder(preprocessor)

        # 5.6.2) Create randomized search object
        # Why RandomizedSearchCV:
        # - cheaper than full grid search
        # - works well when you have many knobs
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=n_iter,
            scoring=TUNING_SCORING,
            cv=cv,
            n_jobs=-1,              # use all CPU cores
            random_state=42,        # reproducible search sampling
            refit=True,             # refit best model on ALL train at the end
            verbose=0,
        )

        # 5.6.3) Fit search using ONLY training data (CV happens inside)
        search.fit(X_train, y_train)

        # 5.6.4) Best pipeline already refit on full train (because refit=True)
        best_model = search.best_estimator_

        # 5.6.5) Save best params for reporting / reproducibility
        best_params_rows.append(
            {
                "model_name": model_name,
                "best_cv_score": search.best_score_,   # this is average_precision on CV
                "best_params": search.best_params_,
            }
        )

        # 5.6.6) Evaluate ONCE on the test set (final holdout evaluation)
        tuned_results = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            threshold=threshold,
        )

        # 5.6.7) Store outputs
        all_results.append(tuned_results)
        results_df = append_results(results_df, tuned_results)
        # Persist this model immediately (atomic write)
        save_model_artifact(tuned_results, ARTIFACT_DIR_Tuned)

    # 5.7) Sort leaderboard (choose your preferred primary metric)
    results_df = sort_results(results_df, by="pr_auc", ascending=False)
    # Persist leaderboard summary
    save_or_update_summary(results_df, ARTIFACT_DIR_Tuned)


    # 5.8) Create best-params table
    best_params_df = pd.DataFrame(best_params_rows)

    return results_df, all_results, best_params_df


# ============================================================
# 6) Script entry point (terminal execution)
# ============================================================

if __name__ == "__main__":
    results_df, _, best_params_df = run_tuned_models(models_to_run=None, threshold=0.5)

    print("\nTuned model results (test set):")
    print(results_df)

    print("\nBest CV params:")
    print(best_params_df)
