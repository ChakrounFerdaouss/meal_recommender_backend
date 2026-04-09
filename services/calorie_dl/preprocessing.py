import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna()

    if df["Gender"].dtype == object:
        df["Gender"] = df["Gender"].map({"male": 1, "female": 0})

    return df