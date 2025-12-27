import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from src.config import DATASET_NAME, TARGET_COLUMN,LABEL_MAPPING,TEST_SIZE,STRATIFY_SPLIT,RANDOM_STATE

def load_dataset():
    """
    Load the German Credit dataset from OpenML.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix containing all input features.
    y : pandas.Series
        Target labels as raw strings (e.g., 'good', 'bad').
        No encoding or transformation is applied here.
    """

    # Fetch dataset from OpenML as a pandas DataFrame
    data = fetch_openml(
        name=DATASET_NAME,
         version=1,
        as_frame=True
    )

    # Extract full DataFrame
    df = data.frame

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y


def encode_target(y: pd.Series) -> pd.Series:
    """
    Encode raw string target labels into numeric values using LABEL_MAPPING.
    """

    # Identify unexpected labels early (fail fast)
    unique_labels = set(y.unique())
    expected_labels = set(LABEL_MAPPING.keys())

    unexpected_labels = unique_labels - expected_labels
    if unexpected_labels:
        raise ValueError(
            f"Unexpected target labels found: {unexpected_labels}. "
            f"Expected only: {expected_labels}"
        )

    # Map string labels to numeric values
    y_encoded = y.map(LABEL_MAPPING)

    return y_encoded

def train_test_split_data(X, y):
    """
    Split features and target into train and test sets.
    """
    stratify = y if STRATIFY_SPLIT else None # We are telling scikit-learn: “Split X and y such that the class proportions in y are preserved in both splits.”

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=stratify,
        random_state=RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test



def get_train_test_data():
    """
    Load the dataset, encode the target, and perform the train/test split.
    """

    # Load raw data
    X, y_raw = load_dataset()

    # Encode target labels
    y_encoded = encode_target(y_raw)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split_data(
        X, y_encoded
    )

    return X_train, X_test, y_train, y_test