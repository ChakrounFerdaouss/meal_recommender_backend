"""
Configuration centrale — hyperparamètres, chemins, mappings.
"""

import os

# ── Chemins ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "artifacts", "model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "scaler.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "artifacts", "eval_report.json")

# ── Source de données (Kaggle en priorité, fallback synthétique) ───────────────
# Dataset Kaggle : "fmendes/fmendesdat263xdemos"
#   exercise.csv  → User_ID, Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp
#   calories.csv  → User_ID, Calories
# 15 000 enregistrements réels
KAGGLE_DATASET = "fmendes/fmendesdat263xdemos"
KAGGLE_FILES   = ["exercise.csv", "calories.csv"]

SYNTHETIC_N    = 25_000   # taille fallback

# ── Features du modèle ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "age",
    "gender",          # 0=female, 1=male
    "weight_kg",
    "height_cm",
    "activity_factor", # 1.2 → 1.9
    "bmr",             # Mifflin-St Jeor
    "bmi",
    "weight_height_ratio",
]
TARGET_COL   = "tdee_kcal"
N_FEATURES   = len(FEATURE_COLS)   # 8

# ── Hyperparamètres ───────────────────────────────────────────────────────────
EPOCHS      = 150
BATCH_SIZE  = 256
LR          = 1e-3
TEST_SIZE   = 0.15
VAL_SIZE    = 0.10
PATIENCE    = 20          # early stopping

# ── Niveaux d'activité ────────────────────────────────────────────────────────
ACTIVITY_MAP: dict[str, float] = {
    "sedentary":   1.200,
    "light":       1.375,
    "moderate":    1.550,
    "active":      1.725,
    "very_active": 1.900,
}
ACTIVITY_INT_MAP: dict[int, float] = {
    1: 1.200, 2: 1.375, 3: 1.550, 4: 1.725, 5: 1.900,
}