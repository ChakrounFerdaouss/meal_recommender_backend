"""
═══════════════════════════════════════════════════════════════
ÉTAPE 6 — TEST ET ÉVALUATION
═══════════════════════════════════════════════════════════════

Métriques calculées sur le jeu de test (jamais vu à l'entraînement) :

  MAE   : Mean Absolute Error               (kcal)
  RMSE  : Root Mean Squared Error           (kcal)
  MAPE  : Mean Absolute Percentage Error    (%)
  R²    : Coefficient de détermination      (0→1)
  R     : Corrélation de Pearson            (0→1)

Interprétation pour un TDEE estimateur :
  MAE < 150 kcal  → excellent
  MAE < 250 kcal  → bon
  R²  > 0.90      → excellent
"""

import json
import os

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import REPORT_PATH


def evaluate(model, scaler, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Évalue le modèle sur le jeu de test et retourne les métriques.
    Sauvegarde aussi le rapport JSON dans REPORT_PATH.
    """
    model.eval()

    X_scaled = scaler.transform(X_test)
    X_tensor  = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_tensor).numpy().flatten()

    y_true = np.array(y_test).flatten()

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)
    r2   = float(r2_score(y_true, y_pred))
    r    = float(np.corrcoef(y_true, y_pred)[0, 1])

    # Distribution des erreurs absolues
    abs_err   = np.abs(y_true - y_pred)
    within_50  = float((abs_err <= 50).mean()  * 100)
    within_100 = float((abs_err <= 100).mean() * 100)
    within_200 = float((abs_err <= 200).mean() * 100)

    metrics = {
        "mae_kcal":        round(mae,  1),
        "rmse_kcal":       round(rmse, 1),
        "mape_pct":        round(mape, 2),
        "r2":              round(r2,   4),
        "pearson_r":       round(r,    4),
        "within_50kcal":   round(within_50,  1),
        "within_100kcal":  round(within_100, 1),
        "within_200kcal":  round(within_200, 1),
        "n_test":          len(y_true),
        "y_mean":          round(float(y_true.mean()), 1),
        "y_std":           round(float(y_true.std()),  1),
    }

    _print_report(metrics)
    _save_report(metrics)

    return metrics


def _print_report(m: dict):
    print("\n" + "═" * 52)
    print("  RAPPORT D'ÉVALUATION — Jeu de test")
    print("═" * 52)
    print(f"  Échantillons testés       : {m['n_test']:,}")
    print(f"  TDEE moyen (vrai)         : {m['y_mean']:.0f} ± {m['y_std']:.0f} kcal")
    print("─" * 52)
    print(f"  MAE                       : {m['mae_kcal']:.1f} kcal")
    print(f"  RMSE                      : {m['rmse_kcal']:.1f} kcal")
    print(f"  MAPE                      : {m['mape_pct']:.2f} %")
    print(f"  R²                        : {m['r2']:.4f}")
    print(f"  Corrélation de Pearson    : {m['pearson_r']:.4f}")
    print("─" * 52)
    print(f"  Préd. dans ±50  kcal      : {m['within_50kcal']:.1f} %")
    print(f"  Préd. dans ±100 kcal      : {m['within_100kcal']:.1f} %")
    print(f"  Préd. dans ±200 kcal      : {m['within_200kcal']:.1f} %")
    quality = (
        "★★★ Excellent" if m["mae_kcal"] < 150 else
        "★★  Bon"       if m["mae_kcal"] < 250 else
        "★   Acceptable"
    )
    print(f"  Qualité globale           : {quality}")
    print("═" * 52 + "\n")


def _save_report(metrics: dict):
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)