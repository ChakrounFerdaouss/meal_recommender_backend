"""
═══════════════════════════════════════════════════════════════
ÉTAPE 3 — REPRÉSENTATION (Feature Engineering)
═══════════════════════════════════════════════════════════════

À partir du DataFrame nettoyé, on construit les 8 features
qui alimenteront le réseau de neurones :

  1. age
  2. gender              (0/1)
  3. weight_kg
  4. height_cm
  5. activity_factor     (inféré depuis calories_raw si absent)
  6. bmr                 (Mifflin-St Jeor)
  7. bmi                 (kg/m²)
  8. weight_height_ratio (feature croisée)

Cible : tdee_kcal (= calories_raw, déjà nettoyée)
"""

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, TARGET_COL, ACTIVITY_INT_MAP


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Construit (X, y) à partir du DataFrame prétraité.

    Retourne :
        X : np.ndarray shape (n, 8)  — features
        y : np.ndarray shape (n,)    — TDEE kcal
    """
    df = df.copy()

    # ── BMR Mifflin-St Jeor ───────────────────────────────────────────────────
    df["bmr"] = (
        10 * df["weight_kg"]
        + 6.25 * df["height_cm"]
        - 5   * df["age"]
        + df["gender"].map({1: 5.0, 0: -161.0})
    )

    # ── BMI ───────────────────────────────────────────────────────────────────
    df["bmi"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2

    # ── Ratio poids/taille ────────────────────────────────────────────────────
    df["weight_height_ratio"] = df["weight_kg"] / df["height_cm"]

    # ── Activity factor ───────────────────────────────────────────────────────
    # Si la colonne n'existe pas (données Kaggle), on l'infère depuis le TDEE
    if "activity_factor" not in df.columns:
        # af = TDEE / BMR, clippé dans [1.2, 1.9]
        df["activity_factor"] = np.clip(
            df["calories_raw"] / df["bmr"].replace(0, np.nan),
            1.2, 1.9
        ).fillna(1.55)

    # ── Cible ─────────────────────────────────────────────────────────────────
    df[TARGET_COL] = df["calories_raw"]

    # ── Sélection finale ──────────────────────────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[features] Colonnes manquantes : {missing}")

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    print(f"[features] ✓ X={X.shape}  y={y.shape}"
          f"  y∈[{y.min():.0f}, {y.max():.0f}] kcal  μ={y.mean():.0f}")
    return X, y


def build_input_vector(age: int, gender: int, weight: float,
                       height: float, activity_factor: float) -> np.ndarray:
    """
    Construit le vecteur de features pour UNE nouvelle observation.
    Utilisé à l'inférence.
    """
    bmr = 10*weight + 6.25*height - 5*age + (5 if gender == 1 else -161)
    bmi = weight / (height / 100) ** 2
    whr = weight / height

    return np.array(
        [[age, gender, weight, height, activity_factor, bmr, bmi, whr]],
        dtype=np.float32
    )