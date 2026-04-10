"""
COUCHE 3 — IA Générative : Recommandations de repas
=====================================================
Compatible refonte :
- CalorieDLService (tdee_kcal)
- ExtractedPreferences (NLP couche 2)
- MealRecommendation schema (FastAPI)
"""

import os
import re
import json
import logging
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)
MODEL = "gpt-4o-mini"

# ─────────────────────────────────────────────
# OpenAI client lazy-safe
# ─────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY manquante → fallback activé")
            api_key = "sk-placeholder"
        _client = OpenAI(api_key=api_key)
    return _client


# ─────────────────────────────────────────────
# TEXT GENERATION (fallback humain)
# ─────────────────────────────────────────────

def generate_meal_plan(
    calories: float,
    preferences: str,
    objectif: str,
    mood: str,
    energy_level: str
) -> str:

    prompt = f"""
Plan repas journalier :

Calories: {calories}
Préférences: {preferences}
Objectif: {objectif}
Humeur: {mood}
Énergie: {energy_level}

Inclure :
- petit-déjeuner
- déjeuner
- dîner
- collations si nécessaire
- conseils motivationnels
"""

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un coach nutritionnel bienveillant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.warning(f"OpenAI fallback text utilisé: {e}")
        return _fallback_text(calories, preferences, objectif, mood, energy_level)


# ─────────────────────────────────────────────
# FALLBACK TEXT
# ─────────────────────────────────────────────

def _fallback_text(calories, preferences, objectif, mood, energy_level):

    return f"""
Plan repas personnalisé :

Petit-déjeuner : yaourt grec + flocons d’avoine + fruits
Déjeuner : riz/quinoa + poulet ou tofu + légumes
Dîner : soupe de légumes + poisson ou légumineuses

Calories ciblées : {calories} kcal
Objectif : {objectif}
Préférences : {preferences}
Humeur : {mood}
Énergie : {energy_level}

Conseil : reste constant et hydrate-toi bien.
"""


# ─────────────────────────────────────────────
# JSON PROMPT (STRUCTURED OUTPUT)
# ─────────────────────────────────────────────

def _build_json_prompt(calories, preferences, objectif, mood, energy_level, meals_per_day, days):

    meal_types = {
        2: ["déjeuner", "dîner"],
        3: ["petit-déjeuner", "déjeuner", "dîner"],
        4: ["petit-déjeuner", "collation matin", "déjeuner", "dîner"],
        5: ["petit-déjeuner", "collation matin", "déjeuner", "collation soir", "dîner"],
        6: ["petit-déjeuner", "collation matin", "déjeuner", "collation après-midi", "dîner", "collation soir"],
    }.get(meals_per_day, ["petit-déjeuner", "déjeuner", "dîner"])

    return f"""
Tu dois répondre UNIQUEMENT en JSON valide.

Créer un plan sur {days} jour(s) :

Calories/jour: {calories}
Préférences: {preferences}
Objectif: {objectif}
Humeur: {mood}
Énergie: {energy_level}

Repas: {", ".join(meal_types)}

Règles:
- Respect calories ±5%
- Adapter humeur + énergie
- Inclure motivation

Format JSON strict:
{{
  "plan": [
    {{
      "day": 1,
      "total_calories": {int(calories)},
      "meals": [
        {{
          "name": "repas",
          "meal_type": "type",
          "calories": 400,
          "proteins_g": 25,
          "carbs_g": 40,
          "fats_g": 15,
          "ingredients": ["..."],
          "instructions": "...",
          "motivation": "..."
        }}
      ]
    }}
  ],
  "weekly_tips": ["...", "..."],
  "hydration_advice": "..."
}}
"""


# ─────────────────────────────────────────────
# MAIN FUNCTION (FASTAPI)
# ─────────────────────────────────────────────

def generate_recommendations(
    calorie_data: dict,
    preferences: dict,
    mood: str = "motivé",
    energy_level: str = "moyen",
    meals_per_day: int = 3,
    days: int = 1
) -> dict:

    # ✔ compat refonte CalorieDLService
    tdee = calorie_data.get("tdee_kcal") or calorie_data.get("tdee")

    goal = preferences.get("goal", "")

    if "perdre" in goal:
        target_kcal = int(tdee * 0.80)
    elif "masse" in goal:
        target_kcal = int(tdee * 1.10)
    else:
        target_kcal = int(tdee)

    # ✔ construction preferences texte
    prefs = []
    if preferences.get("diet_type"):
        prefs.append(preferences["diet_type"])
    if preferences.get("cuisine_style"):
        prefs.append("cuisine " + "/".join(preferences["cuisine_style"]))
    if preferences.get("restrictions"):
        prefs.append("sans " + ", ".join(preferences["restrictions"]))
    if preferences.get("preferred_proteins"):
        prefs.append("protéines: " + ", ".join(preferences["preferred_proteins"]))

    preferences_str = " | ".join(prefs) if prefs else "équilibré"

    logger.info("Génération plan repas IA")

    try:
        response = _get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Répond uniquement en JSON valide."},
                {
                    "role": "user",
                    "content": _build_json_prompt(
                        target_kcal,
                        preferences_str,
                        goal or "maintenir santé",
                        mood,
                        energy_level,
                        meals_per_day,
                        days
                    )
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )

        raw = response.choices[0].message.content.strip()

        # cleanup JSON
        raw = re.sub(r"```json|```", "", raw).strip()

        data = json.loads(raw)
        data["model_used"] = f"OpenAI {MODEL}"
        return data

    except Exception as e:
        logger.warning(f"Fallback structuré activé: {e}")

        return _build_fallback_structured(
            target_kcal,
            meals_per_day,
            days
        )


# ─────────────────────────────────────────────
# FALLBACK STRUCTURED (FIXED)
# ─────────────────────────────────────────────

def _build_fallback_structured(calories, meals_per_day, days):

    per_meal = calories // meals_per_day

    meals = [
        {
            "name": "Repas équilibré",
            "meal_type": "standard",
            "calories": per_meal,
            "proteins_g": round(per_meal * 0.25 / 4, 1),
            "carbs_g": round(per_meal * 0.45 / 4, 1),
            "fats_g": round(per_meal * 0.30 / 9, 1),
            "ingredients": ["légumes", "protéine", "féculent"],
            "instructions": "Préparer simplement et équilibré",
            "motivation": "Continue tes efforts 💪"
        }
        for _ in range(meals_per_day)
    ]

    return {
        "plan": [
            {
                "day": i + 1,
                "total_calories": calories,
                "meals": meals
            }
            for i in range(days)
        ],
        "weekly_tips": [
            "Hydrate-toi bien",
            "Mange équilibré",
            "Reste constant"
        ],
        "hydration_advice": "1.5L à 2L d'eau par jour",
        "model_used": "fallback_structured"
    }