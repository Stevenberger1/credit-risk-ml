from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def build_xgboost_pipeline(preprocessor):
    """
    Build a baseline XGBoost classification pipeline.
    """

    xgb = XGBClassifier(
        n_estimators=300,          # Reasonable baseline size
        max_depth=6,               # Default-ish depth
        learning_rate=0.1,         # Standard baseline LR
        subsample=0.8,             # Row subsampling
        colsample_bytree=0.8,      # Feature subsampling
        objective="binary:logistic", # outputs probabilities (needed for ROC / PR)
        eval_metric="logloss",     # Required to silence warnings
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", xgb),
        ]
    )

    return pipeline
