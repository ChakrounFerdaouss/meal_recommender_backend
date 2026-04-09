from datasets import load_dataset
import pandas as pd
from .config import DATASET_ID


def load_data():
    """
    Charge dataset Hugging Face ou fallback synthétique
    """
    try:
        ds = load_dataset(DATASET_ID, split="train")
        return ds.to_pandas()
    except Exception:
        return _synthetic()


def _synthetic(n=2000):
    import numpy as np

    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "Age": rng.integers(18, 70, n),
        "Gender": rng.choice([0, 1], n),
        "Weight": rng.uniform(45, 120, n),
        "Height": rng.uniform(150, 200, n),
        "PhysicalActivityLevel": rng.uniform(1.2, 1.9, n),
    })

    df["Calories"] = (
        10 * df["Weight"] + 6.25 * df["Height"] - 5 * df["Age"]
        + np.where(df["Gender"] == 1, 5, -161)
    ) * df["PhysicalActivityLevel"]

    return df