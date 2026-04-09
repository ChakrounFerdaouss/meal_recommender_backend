import numpy as np


def compute_bmr(df):
    return np.where(
        df["Gender"] == 1,
        10 * df["Weight"] + 6.25 * df["Height"] - 5 * df["Age"] + 5,
        10 * df["Weight"] + 6.25 * df["Height"] - 5 * df["Age"] - 161
    )


def build_features(df):
    df = df.copy()

    df["BMR"] = compute_bmr(df)

    X = df[[
        "Age",
        "Gender",
        "Weight",
        "Height",
        "PhysicalActivityLevel",
        "BMR"
    ]].values

    y = df["Calories"].values

    return X, y