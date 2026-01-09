import os
import pickle
import pandas as pd


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_model_artifact(result: dict, output_dir: str):
    """
    Save a single model's full evaluation artifact as a pickle.
    """
    ensure_dir(output_dir)

    model_key = result["model_name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
    path = os.path.join(output_dir, f"{model_key}.pkl")

    with open(path, "wb") as f:
        pickle.dump(result, f)


def load_model_artifact(path: str) -> dict:
    """Load a single model artifact."""
    with open(path, "rb") as f:
        return pickle.load(f)


def save_or_update_summary(results_df: pd.DataFrame, output_dir: str):
    """
    Save (overwrite) summary table.
    """
    ensure_dir(output_dir)
    results_df.to_csv(os.path.join(output_dir, "results_summary.csv"), index=False)


def load_summary(output_dir: str) -> pd.DataFrame:
    """Load summary table."""
    return pd.read_csv(os.path.join(output_dir, "results_summary.csv"))
