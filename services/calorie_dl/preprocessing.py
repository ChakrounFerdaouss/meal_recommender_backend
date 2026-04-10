"""
═══════════════════════════════════════════════════════════════
ÉTAPE 2 — PRÉTRAITEMENTS
═══════════════════════════════════════════════════════════════

  - Nettoyage (NaN, doublons, outliers)
  - Encodage Gender → 0/1
  - Normalisation Height cm ↔ m
  - Filtrage valeurs aberrantes physiologiques
  - Rapport de qualité loggé
"""

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et normalise le DataFrame brut (format Kaggle ou synthétique).
    Retourne un DataFrame propre avec des colonnes standardisées.
    """
    df = df.copy()
    n_raw = len(df)
    print(f"[preprocess] Entrée : {n_raw:,} lignes")

    # ── 1. Supprime les doublons ───────────────────────────────────────────────
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "User_ID"])
    print(f"[preprocess] Après dédoublonnage : {len(df):,} lignes")

    # ── 2. Supprime les NaN ───────────────────────────────────────────────────
    df = df.dropna()
    print(f"[preprocess] Après dropna : {len(df):,} lignes")

    # ── 3. Encodage Gender ────────────────────────────────────────────────────
    # Kaggle : "male" / "female"  |  synthétique : idem
    if "Gender" in df.columns and df["Gender"].dtype == object:
        df["Gender"] = (
            df["Gender"].str.lower().str.strip()
            .map({"male": 1, "female": 0, "m": 1, "f": 0, "homme": 1, "femme": 0})
        )

    # ── 4. Hauteur : convertit m → cm si nécessaire (Kaggle = mètres) ─────────
    if "Height" in df.columns and df["Height"].median() < 10:
        df["Height"] = df["Height"] * 100
        print("[preprocess] Height converti m → cm")

    # ── 5. Renommage vers colonnes internes ───────────────────────────────────
    rename = {
        "Age":        "age",
        "Gender":     "gender",
        "Weight":     "weight_kg",
        "Height":     "height_cm",
        "Heart_Rate": "heart_rate",
        "Duration":   "duration_min",
        "Body_Temp":  "body_temp_c",
        "Calories":   "calories_raw",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # ── 6. Filtrage outliers physiologiques ───────────────────────────────────
    filters = {
        "age":         (15, 90),
        "weight_kg":   (30, 200),
        "height_cm":   (140, 215),
        "calories_raw":(800, 7000),
    }
    for col, (lo, hi) in filters.items():
        if col in df.columns:
            before = len(df)
            df = df[(df[col] >= lo) & (df[col] <= hi)]
            removed = before - len(df)
            if removed > 0:
                print(f"[preprocess] Filtre {col} [{lo},{hi}] → {removed} lignes supprimées")

    # ── 7. Cast types ─────────────────────────────────────────────────────────
    float_cols = ["age", "weight_kg", "height_cm", "calories_raw"]
    for c in float_cols:
        if c in df.columns:
            df[c] = df[c].astype(float)

    df = df.reset_index(drop=True)
    print(f"[preprocess] ✓ Sortie : {len(df):,} lignes propres "
          f"({n_raw - len(df):,} supprimées soit "
          f"{(n_raw - len(df)) / n_raw * 100:.1f}%)")
    return df