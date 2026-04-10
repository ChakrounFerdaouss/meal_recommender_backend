"""
═══════════════════════════════════════════════════════════════
ÉTAPE 1 — COLLECTE DES DONNÉES
═══════════════════════════════════════════════════════════════

Source prioritaire : Kaggle "fmendes/fmendesdat263xdemos"
  → exercise.csv  (User_ID, Gender, Age, Height, Weight,
                   Duration, Heart_Rate, Body_Temp)
  → calories.csv  (User_ID, Calories)
  → 15 000 enregistrements réels de séances de sport

Fallback : dataset synthétique Mifflin-St Jeor (25 000 profils)
  → utilisé si Kaggle est inaccessible (env. sans internet)
"""

import numpy as np
import pandas as pd

from .config import KAGGLE_DATASET, SYNTHETIC_N, ACTIVITY_INT_MAP


# ── Entrée publique ───────────────────────────────────────────────────────────

def collect_data() -> tuple[pd.DataFrame, str]:
    """
    Retourne (DataFrame brut, source_label).
    Le DataFrame contient les colonnes RAW, non transformées.
    """
    df, source = _try_kaggle()
    if df is None:
        print("[dataset] ⚠ Kaggle inaccessible → fallback synthétique")
        df     = _synthetic(SYNTHETIC_N)
        source = f"synthetic_mifflin (n={SYNTHETIC_N:,})"
    else:
        print(f"[dataset] ✓ Kaggle chargé : {len(df):,} lignes")
    return df, source


# ── Kaggle ────────────────────────────────────────────────────────────────────

def _try_kaggle() -> tuple[pd.DataFrame | None, str]:
    """
    Télécharge exercise.csv + calories.csv depuis Kaggle et les fusionne.
    Retourne (None, "") si le téléchargement échoue.
    """
    try:
        import kagglehub, os

        path = kagglehub.dataset_download(KAGGLE_DATASET)

        ex_path  = os.path.join(path, "exercise.csv")
        cal_path = os.path.join(path, "calories.csv")

        if not os.path.exists(ex_path) or not os.path.exists(cal_path):
            return None, ""

        exercise = pd.read_csv(ex_path)
        calories = pd.read_csv(cal_path)

        # Jointure sur User_ID
        df = exercise.merge(calories, on="User_ID", how="inner")

        return df, f"kaggle/{KAGGLE_DATASET} (n={len(df):,})"

    except Exception as e:
        print(f"[dataset] Kaggle error: {e}")
        return None, ""


# ── Fallback synthétique ──────────────────────────────────────────────────────

def _synthetic(n: int) -> pd.DataFrame:
    """
    Génère n profils réalistes via Mifflin-St Jeor + Harris-Benedict.

    Colonnes produites (format identique à Kaggle après fusion) :
      Gender, Age, Height, Weight, Heart_Rate, Duration,
      Body_Temp, Calories  ← target proxy TDEE
    """
    rng = np.random.default_rng(42)

    gender  = rng.choice(["male", "female"], n, p=[0.50, 0.50])
    age     = rng.integers(16, 80, n).astype(float)
    weight  = rng.uniform(44, 140, n)    # kg
    height  = rng.uniform(150, 205, n)   # cm

    # Fréquence cardiaque repos 55-85, monte à l'effort
    heart_rate  = rng.integers(55, 180, n).astype(float)
    # Durée exercice 10-120 min
    duration    = rng.uniform(10, 120, n)
    # Température corporelle 36.5-40.5 °C
    body_temp   = rng.uniform(36.5, 40.5, n)

    # TDEE Mifflin-St Jeor
    is_male = (gender == "male").astype(float)
    bmr     = 10*weight + 6.25*height - 5*age + np.where(is_male, 5, -161)
    act_factors = np.array(list(ACTIVITY_INT_MAP.values()))
    act_probs   = np.array([0.25, 0.30, 0.25, 0.12, 0.08])
    af = rng.choice(act_factors, n, p=act_probs)

    # Calories = TDEE + bruit biologique ±8 %
    calories = np.clip(bmr * af * rng.normal(1.0, 0.08, n), 1200, 6500)

    return pd.DataFrame({
        "User_ID":    np.arange(1, n + 1),
        "Gender":     gender,
        "Age":        age,
        "Height":     height,
        "Weight":     weight,
        "Heart_Rate": heart_rate,
        "Duration":   duration,
        "Body_Temp":  body_temp,
        "Calories":   calories,
    })