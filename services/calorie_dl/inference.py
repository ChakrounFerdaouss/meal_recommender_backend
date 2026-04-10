"""
═══════════════════════════════════════════════════════════════
ÉTAPE 7 — TEST SUR UNE NOUVELLE DONNÉE
═══════════════════════════════════════════════════════════════

Transforme les inputs bruts de l'utilisateur en prédiction TDEE.

Accepte tous les formats d'entrée :
  gender   : "male" | "female" | "m" | "f" | 0 | 1
  activity : "moderate" | 3 | 1.55
"""

import numpy as np
import torch

from .config import ACTIVITY_MAP, ACTIVITY_INT_MAP
from .features import build_input_vector


# ── Résolution des inputs ─────────────────────────────────────────────────────

def resolve_gender(gender) -> int:
    if isinstance(gender, (int, float)):
        return int(gender)
    g = str(gender).lower().strip()
    mapping = {"male": 1, "m": 1, "homme": 1, "h": 1,
               "female": 0, "f": 0, "femme": 0}
    if g not in mapping:
        raise ValueError(
            f"Genre inconnu : '{gender}'. Attendu : male/female"
        )
    return mapping[g]


def resolve_activity(activity) -> float:
    # String label
    if isinstance(activity, str):
        key = activity.lower().strip()
        if key not in ACTIVITY_MAP:
            raise ValueError(
                f"Niveau d'activité inconnu : '{activity}'. "
                f"Valeurs : {list(ACTIVITY_MAP.keys())}"
            )
        return ACTIVITY_MAP[key]

    val = float(activity)

    # Entier 1-5
    if 1 <= val <= 5 and val == int(val):
        return ACTIVITY_INT_MAP[int(val)]

    # Facteur flottant direct
    if 1.0 <= val <= 2.5:
        return val

    raise ValueError(
        f"Activité invalide : {activity}. "
        "Attendu : 'moderate' | entier 1-5 | flottant 1.2-1.9"
    )


# ── Inférence ─────────────────────────────────────────────────────────────────

def predict_one(model, scaler,
                age: int, gender, weight: float,
                height: float, activity) -> float:
    """
    Prédit le TDEE pour UN profil utilisateur.

    Args:
        model    : CalorieResNet entraîné
        scaler   : StandardScaler fitté sur X_train
        age      : âge en années
        gender   : "male"/"female" ou 0/1
        weight   : poids kg
        height   : taille cm
        activity : niveau d'activité

    Returns:
        TDEE prédit en kcal (float, clippé dans [800, 7000])
    """
    g  = resolve_gender(gender)
    af = resolve_activity(activity)

    x        = build_input_vector(age, g, weight, height, af)
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        raw = float(model(x_tensor).item())

    return float(np.clip(raw, 800.0, 7000.0))


def run_sample_tests(model, scaler) -> list[dict]:
    """
    Lance 6 cas de référence documentés pour valider le modèle.
    Chaque cas inclut le TDEE théorique Mifflin-St Jeor pour comparaison.
    """
    cases = [
        # (label, age, gender, weight, height, activity, tdee_theorique)
        ("Femme 28 ans / moderate",   28, "female", 62.0,  166.0, "moderate",   2102),
        ("Homme 35 ans / active",     35, "male",   80.0,  178.0, "active",     3050),
        ("Femme 55 ans / sedentary",  55, "female", 70.0,  162.0, "sedentary",  1559),
        ("Homme 22 ans / very_active",22, "male",   75.0,  178.0, "very_active",3344),
        ("Homme 45 ans / light",      45, "male",   95.0,  175.0, "light",      2617),
        ("Femme 30 ans / active",     30, "female", 58.0,  170.0, "active",     2449),
    ]

    print("\n" + "═" * 66)
    print("  ÉTAPE 7 — TEST SUR NOUVELLES DONNÉES")
    print("═" * 66)
    print(f"  {'Profil':<34} {'Théorique':>10} {'Prédit':>10} {'Écart':>8}")
    print("─" * 66)

    results = []
    for label, age, gender, weight, height, activity, theorique in cases:
        predicted = predict_one(model, scaler, age, gender, weight, height, activity)
        ecart     = predicted - theorique
        sign      = "+" if ecart >= 0 else ""
        print(f"  {label:<34} {theorique:>9.0f}  {predicted:>9.0f}  {sign}{ecart:>6.0f}")
        results.append({
            "profil":           label,
            "tdee_theorique":   theorique,
            "tdee_predit":      round(predicted, 1),
            "ecart_kcal":       round(ecart, 1),
            "ecart_pct":        round(abs(ecart) / theorique * 100, 1),
        })

    avg_ecart = np.mean([abs(r["ecart_kcal"]) for r in results])
    print("─" * 66)
    print(f"  Écart absolu moyen : {avg_ecart:.0f} kcal")
    print("═" * 66 + "\n")

    return results