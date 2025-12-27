"""
Feature preprocessing utilities for the Credit Risk ML project.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import NUMERIC_IMPUTATION_STRATEGY,CATEGORICAL_IMPUTATION_STRATEGY
from sklearn.compose import ColumnTransformer


def get_feature_types(X_train: pd.DataFrame):
    """
    Identify numeric and categorical feature columns based on pandas dtypes.
    """

    if not isinstance(X_train, pd.DataFrame): #  built-in Python function used to check if an object is an instance of a specified class or a subclass thereof.
        raise TypeError(
            "X_train must be a pandas DataFrame. "
            f"Received type: {type(X_train)}"
        )

    # Numeric features: integers and floats
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Categorical features: object and pandas categorical dtype
    categorical_features = X_train.select_dtypes( 
        include=["object", "category","bool"] # In pandas, strings are stored as object dtype.
        #If someone gives you data, a categorical column might be stored either as object (strings) or as category, and in your ML pipeline, both are treated as categorical features.
        
    ).columns.tolist()

    return numeric_features, categorical_features



def build_numeric_pipeline():
    """
    Build the preprocessing pipeline for numeric features.

    The pipeline performs:
    1. Imputation using the median (robust to outliers)
    2. Standard scaling (mean=0, std=1)

    """

    numeric_pipeline = Pipeline(  # Pipeline enforces correct order (first imputation, then scaling)
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=NUMERIC_IMPUTATION_STRATEGY),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    return numeric_pipeline

def build_categorical_pipeline():
    """
    Build the preprocessing pipeline for categorical features.

    The pipeline performs:
    1. Imputation using the most frequent value
    2. One-hot encoding with safe handling of unseen categories
    """

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy=CATEGORICAL_IMPUTATION_STRATEGY),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", # Test data may contain categories never seen in training
                    sparse_output=False,
                ),
# When sparse_output = False (Dense): returns a NumPy array
# When sparse_output = True (Sparse): returns a sparse matrix

            ),
        ]
    )

    return categorical_pipeline


def build_preprocessor(X_train):
    """
    Build the full preprocessing transformer for the dataset.

    This function:
    - Detects numeric and categorical feature columns from training data
    - Builds separate preprocessing pipelines for each feature type
    - Combines them using a ColumnTransformer
    """

    # Identify feature types from training data
    numeric_features, categorical_features = get_feature_types(X_train)

    # Build individual pipelines
    numeric_pipeline = build_numeric_pipeline()
    categorical_pipeline = build_categorical_pipeline()

    # Combine pipelines into a ColumnTransformer
    preprocessor = ColumnTransformer( # Applies different preprocessing to different columns
        transformers=[
            ("num", numeric_pipeline, numeric_features), # 'num' is just a name for the transformer
            ("cat", categorical_pipeline, categorical_features),# 'cat' is just a name for the transformer
        ],
        remainder="drop", # Drop any columns not specified in transformers
    )

    return preprocessor


'''
What if __name__ == "__main__" is actually for

That block means:

“Run this code only when this file is executed directly, not when imported.”

It is not meant to define the program entry point of a system with multiple components.
'''