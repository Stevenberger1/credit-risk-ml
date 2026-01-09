from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def build_lightgbm_pipeline(preprocessor):
    """
    Build a baseline LightGBM classification pipeline.
    """

    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,              # Let trees grow freely (baseline)
        num_leaves=31,             # Default LightGBM value
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",   # Important for credit risk
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", lgbm),
        ]
    )

    return pipeline
