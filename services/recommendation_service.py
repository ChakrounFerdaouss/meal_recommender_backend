"""
COUCHE 3 — IA Générative : Recommandations de repas via OpenAI
==============================================================
Reproduit fidèlement la structure du code fourni :
  - client OpenAI avec clé depuis .env
  - generate_meal_plan(calories, preferences, objectif, mood, energy_level)
  - Fallback texte si l'API est indisponible
  - Wrapping FastAPI : génère en plus un JSON structuré pour l'API

Workflow :
  1. Construit le prompt depuis les données couches 1 + 2 + mood + energy_level
  2. Appel gpt-4o-mini (temperature=0.7, max_tokens=1500)
  3. Parse la réponse texte → structure JSON pour les routes FastAPI
  4. Fallback automatique si quota/clé absente
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

# ─── Client OpenAI — lazy init pour éviter l erreur au démarrage sans clé ───

_openai_client = None

def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "")
        _openai_client = OpenAI(api_key=api_key or "sk-placeholder")
    return _openai_client

# Proxy pour les mocks pytest (patch services.recommendation_service.client.chat...)
class _ClientProxy:
    class _Chat:
        class _Completions:
            @staticmethod
            def create(**kw):
                return _get_client().chat.completions.create(**kw)
        completions = _Completions()
    chat = _Chat()

client = _ClientProxy()


# ─── Fonction principale (votre structure exacte) ────────────────────────────

def generate_meal_plan(
    calories: float,
    preferences: str,
    objectif: str,
    mood: str,
    energy_level: str
) -> str:
    """
    Génère un plan repas en texte clair et motivant.
    Signature identique à votre code original.
    Fallback automatique si l'API OpenAI est indisponible.
    """
    prompt = f"""
    Crée un plan repas journalier adapté aux besoins suivants :
    - Calories : {calories} kcal
    - Préférences : {preferences}
    - Objectif : {objectif}
    - Humeur : {mood}
    - Niveau d'énergie : {energy_level}

    Inclure :
    - Petit-déjeuner
    - Déjeuner
    - Dîner
    - Collations si nécessaire
    Ajouter des conseils motivants pour atteindre l'objectif.
    Rédige le plan de façon claire et facile à suivre.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un coach nutritionnel bienveillant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.warning(f"OpenAI indisponible ({e}), utilisation du fallback")
        return _fallback_plan(calories, preferences, objectif, mood, energy_level)


# ─── Fallback (votre fallback exact, enrichi) ────────────────────────────────

def _fallback_plan(
    calories: float,
    preferences: str,
    objectif: str,
    mood: str,
    energy_level: str
) -> str:
    """Plan de repas par défaut si l'API OpenAI est indisponible."""

    # Adapte les suggestions selon l'humeur et l'énergie
    breakfast_boost = (
        "smoothie protéiné banane-épinards + flocons d'avoine"
        if energy_level in ("très faible", "faible")
        else "yaourt grec + flocons d'avoine + fruits rouges"
    )
    comfort_note = (
        " (repas réconfortant adapté à votre humeur 💙)"
        if mood in ("stressé", "déprimé", "anxieux")
        else ""
    )

    return f"""
    Plan repas personnalisé{comfort_note} :

    🌅 Petit-déjeuner : {breakfast_boost}
    🥗 Déjeuner : salade quinoa, légumes grillés, poulet ou tofu
    🍎 Collation : amandes + fruit de saison
    🍲 Dîner : soupe de légumes + poisson ou lentilles

    ✨ Conseils motivants :
    - Visez environ {calories} kcal/jour — vous êtes sur la bonne voie !
    - Objectif : {objectif} — chaque repas est un pas vers votre but.
    - Humeur du jour : {mood} — l'alimentation influence positivement l'humeur.
    - Énergie actuelle : {energy_level} — ce plan est adapté à votre niveau.
    - Préférences : {preferences}
    - Hydratez-vous bien : 1,5 à 2L d'eau par jour.
    """


# ─── Version structurée JSON pour l'API FastAPI ──────────────────────────────

def _build_json_prompt(
    calories: float,
    preferences: str,
    objectif: str,
    mood: str,
    energy_level: str,
    meals_per_day: int,
    days: int
) -> str:
    """
    Prompt étendu demandant une réponse JSON structurée.
    Utilisé par generate_recommendations() pour l'API REST.
    """
    meal_types = {
        2: ["déjeuner", "dîner"],
        3: ["petit-déjeuner", "déjeuner", "dîner"],
        4: ["petit-déjeuner", "déjeuner", "collation", "dîner"],
        5: ["petit-déjeuner", "collation matin", "déjeuner", "collation après-midi", "dîner"],
        6: ["petit-déjeuner", "collation matin", "déjeuner", "collation après-midi", "dîner", "collation soir"],
    }.get(meals_per_day, ["petit-déjeuner", "déjeuner", "dîner"])

    return f"""
    Crée un plan repas pour {days} jour(s) adapté aux besoins suivants :
    - Calories : {calories} kcal/jour
    - Préférences alimentaires : {preferences}
    - Objectif : {objectif}
    - Humeur actuelle : {mood}
    - Niveau d'énergie : {energy_level}
    - Repas par jour : {meals_per_day} ({', '.join(meal_types)})

    Règles importantes :
    - Respecte les {calories} kcal (±5%)
    - Adapte les repas à l'humeur "{mood}" (ex: si stressé → aliments anti-stress)
    - Adapte les repas au niveau d'énergie "{energy_level}" (ex: si faible → repas énergisants)
    - Inclure un message motivant et personnalisé pour chaque repas
    - Ajouter des conseils motivants pour atteindre l'objectif

    Réponds UNIQUEMENT avec ce JSON valide, sans markdown ni texte hors JSON :

    {{
      "plan": [
        {{
          "day": 1,
          "total_calories": {int(calories)},
          "meals": [
            {{
              "name": "Nom du repas",
              "meal_type": "petit-déjeuner",
              "calories": 400,
              "proteins_g": 20.0,
              "carbs_g": 50.0,
              "fats_g": 12.0,
              "ingredients": ["ingrédient 1", "ingrédient 2"],
              "instructions": "Instructions simples en 2-3 phrases.",
              "motivation": "Message motivant personnalisé lié à l'humeur et à l'objectif."
            }}
          ]
        }}
      ],
      "weekly_tips": [
        "Conseil motivant 1 lié à l'objectif",
        "Conseil pratique 2",
        "Conseil bien-être 3"
      ],
      "hydration_advice": "Conseil hydratation personnalisé selon l'énergie et l'humeur."
    }}
    """


def generate_recommendations(
    calorie_data: dict,
    preferences: dict,
    mood: str = "motivé",
    energy_level: str = "moyen",
    meals_per_day: int = 3,
    days: int = 1
) -> dict:
    """
    Point d'entrée FastAPI — retourne un dict JSON structuré.
    Utilise generate_meal_plan() en interne puis parse en JSON.
    """
    # Calcul des calories cibles selon l'objectif (cohérence avec couche 1)
    tdee   = calorie_data["tdee"]
    goal   = preferences.get("goal", "")
    if "perdre" in goal:
        target_kcal = round(tdee * 0.80)
    elif "masse" in goal or "musculaire" in goal:
        target_kcal = round(tdee * 1.10)
    else:
        target_kcal = round(tdee)

    # Construction des préférences en texte lisible
    prefs_parts = []
    if preferences.get("diet_type"):
        prefs_parts.append(preferences["diet_type"])
    if preferences.get("cuisine_style"):
        prefs_parts.append("cuisine " + "/".join(preferences["cuisine_style"]))
    if preferences.get("restrictions"):
        prefs_parts.append("sans " + ", ".join(preferences["restrictions"]))
    if preferences.get("preferred_proteins"):
        prefs_parts.append("protéines : " + ", ".join(preferences["preferred_proteins"]))
    preferences_str = " | ".join(prefs_parts) if prefs_parts else "équilibré"

    logger.info(f"Appel OpenAI — {days}j, {meals_per_day} repas, humeur={mood}, énergie={energy_level}")

    # Essai avec prompt JSON structuré
    json_prompt = _build_json_prompt(
        calories=target_kcal,
        preferences=preferences_str,
        objectif=goal or "maintenir la santé",
        mood=mood,
        energy_level=energy_level,
        meals_per_day=meals_per_day,
        days=days
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Tu es un coach nutritionnel bienveillant. Réponds uniquement en JSON valide."},
                {"role": "user",   "content": json_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        raw = response.choices[0].message.content.strip()

        # Nettoyage éventuel de blocs markdown
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw.strip())

        plan = json.loads(raw)
        plan["model_used"] = f"OpenAI {MODEL}"
        logger.info(f"Plan JSON généré avec succès — {len(plan.get('plan', []))} jour(s)")
        return plan

    except Exception as e:
        # Fallback : génère le texte simple puis le structure manuellement
        logger.warning(f"JSON OpenAI échoué ({e}), utilisation du fallback structuré")
        fallback_text = _fallback_plan(
            calories=target_kcal,
            preferences=preferences_str,
            objectif=goal or "maintenir la santé",
            mood=mood,
            energy_level=energy_level
        )
        return _text_to_structured(fallback_text, target_kcal, meals_per_day, days)


def _text_to_structured(text: str, calories: int, meals_per_day: int, days: int) -> dict:
    """Convertit le plan texte fallback en structure JSON compatible avec l'API."""
    per_meal = calories // meals_per_day
    meal_names = {
        "petit-déjeuner": ("Petit-déjeuner équilibré",     per_meal),
        "déjeuner":       ("Déjeuner complet",             per_meal),
        "collation":      ("Collation saine",              per_meal),
        "dîner":          ("Dîner léger et nutritif",      per_meal),
    }
    meal_keys = {
        2: ["déjeuner", "dîner"],
        3: ["petit-déjeuner", "déjeuner", "dîner"],
        4: ["petit-déjeuner", "déjeuner", "collation", "dîner"],
    }.get(meals_per_day, ["petit-déjeuner", "déjeuner", "dîner"])

    meals = [
        {
            "name":         meal_names[k][0],
            "meal_type":    k,
            "calories":     meal_names[k][1],
            "proteins_g":   round(meal_names[k][1] * 0.25 / 4, 1),
            "carbs_g":      round(meal_names[k][1] * 0.45 / 4, 1),
            "fats_g":       round(meal_names[k][1] * 0.30 / 9, 1),
            "ingredients":  ["Ingrédients frais de saison"],
            "instructions": "Préparez avec soin et savourez chaque bouchée.",
            "motivation":   "Vous faites un excellent choix pour votre santé !"
        }
        for k in meal_keys
    ]

    return {
        "plan": [{"day": d + 1, "total_calories": calories, "meals": meals} for d in range(days)],
        "weekly_tips": [
            "Préparez vos repas à l'avance pour rester sur la bonne voie.",
            "Mangez lentement et savourez chaque repas.",
            "Écoutez votre corps — il sait ce dont il a besoin."
        ],
        "hydration_advice": "Buvez au moins 1,5 à 2L d'eau par jour, surtout avant les repas.",
        "model_used": f"Fallback structuré (OpenAI {MODEL} indisponible)"
    }