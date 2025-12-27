"""
Global configuration file for the Credit Risk ML project.

This file centralizes all fixed design decisions:
- dataset identity
- target definition
- splitting strategy
- evaluation rules
- reproducibility settings
- output paths
Each configuration variable is documented for clarity.
"""

# ==============================
# Reproducibility
# ==============================
RANDOM_STATE = 42
"""
Global random seed used across the entire project.
Ensures reproducible splits, model training, and comparisons.
"""

# ==============================
# Dataset configuration
# ==============================
DATASET_NAME = "credit-g"
TARGET_COLUMN = "class"

# ==============================
# Target definition
# ==============================
POSITIVE_CLASS_LABEL = "bad"  # The label treated as the positive class.
NEGATIVE_CLASS_LABEL = "good"  # The label treated as the negative class.

LABEL_MAPPING = { # explicit mapping (used later for clarity)
    NEGATIVE_CLASS_LABEL: 0,
    POSITIVE_CLASS_LABEL: 1,
}


# ==============================
# Train / test split configuration
# ==============================
TEST_SIZE = 0.2 # Proportion of the dataset reserved for the final test set.
STRATIFY_SPLIT = True # Whether to preserve class proportions during the train/test split.


# ==============================
# Cross-validation configuration
# ==============================
CV_FOLDS = 5 # Number of folds for cross-validation on the training set.
CV_STRATEGY = "stratified" #Cross-validation strategy , Used to ensure consistent handling of class imbalance


# ==============================
# Preprocessing configuration
# ==============================
NUMERIC_IMPUTATION_STRATEGY = "median" # Imputation strategy for numerical features, Median is robust to outliers.
CATEGORICAL_IMPUTATION_STRATEGY = "most_frequent" # Imputation strategy for categorical features.
ONEHOT_HANDLE_UNKNOWN = "ignore" # How to handle unseen categories during one-hot encoding.

# ==============================
# Model configuration
# ==============================
MODEL_NAMES = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]
"""
Canonical list of models to be trained and compared.
Used to drive training loops and evaluation.
"""

# ==============================
# Evaluation configuration
# ==============================
EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]
"""
Metrics reported for all models.
Accuracy is included for completeness but not used alone.
"""

PRIMARY_MODEL_SELECTION_METRIC = "pr_auc"
"""
Primary metric used for comparing and selecting models.
Chosen due to class imbalance and focus on positive class detection.
"""

DEFAULT_DECISION_THRESHOLD = 0.5
"""
Default classification threshold.
Will be tuned explicitly during evaluation.
"""

# ==============================
# Output paths
# ==============================

OUTPUT_DIR = "outputs"
MODELS_DIR = f"{OUTPUT_DIR}/models"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
REPORTS_DIR = f"{OUTPUT_DIR}/reports"
"""
Centralized output directories for artifacts.
"""
