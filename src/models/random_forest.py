from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def build_random_forest_pipeline(preprocessor):
    """
    Build a baseline Random Forest classification pipeline.

    This function defines a non-tuned Random Forest model
    combined with the shared preprocessing pipeline.
    """

    random_forest = RandomForestClassifier(
        n_estimators=200,        # Number of trees in the forest
        max_depth=None,          # Let trees grow fully (baseline)
        min_samples_split=2,     # Minimum samples to split a node
        min_samples_leaf=1,      # Minimum samples in a leaf
        class_weight="balanced", # Handle class imbalance
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", random_forest),
        ]
    )

    return pipeline
