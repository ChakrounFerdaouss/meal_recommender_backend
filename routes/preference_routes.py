"""
COUCHE 2 — IA NLP : Analyse des préférences utilisateur
=========================================================
Modèle : facebook/bart-large-mnli (zero-shot classification)
  → Pas de fine-tuning nécessaire
  → Détecte automatiquement : régime, objectif, cuisine, restrictions

Workflow :
  1. Nettoyage du texte
  2. Zero-shot classification sur plusieurs dimensions
  3. Agrégation et structuration des résultats
"""

# routes/preference_routes.py
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import logging
import re

logger = logging.getLogger(__name__)

# ─── Définition de la couche NLP (même code que layer2_nlp) ───────────────

DIET_LABELS = ["végétarien", "vegan", "omnivore", "keto", "méditerranéen",
               "paléo", "sans gluten", "halal", "kasher", "flexitarien"]
GOAL_LABELS = ["perdre du poids", "prendre de la masse musculaire",
               "maintenir son poids", "améliorer sa santé",
               "booster son énergie", "contrôler son diabète", "réduire le cholestérol"]
CUISINE_LABELS = ["cuisine française", "cuisine asiatique", "cuisine méditerranéenne",
                  "cuisine italienne", "cuisine mexicaine", "cuisine indienne",
                  "cuisine japonaise", "cuisine africaine", "cuisine américaine"]
RESTRICTION_LABELS = ["sans lactose", "sans gluten", "sans noix",
                      "sans fruits de mer", "sans porc", "sans alcool",
                      "faible en sodium", "faible en sucre"]
PROTEIN_LABELS = ["poulet", "bœuf", "poisson", "légumineuses", "tofu",
                  "œufs", "fruits de mer", "dinde", "agneau"]

PIPELINE_MODEL = "facebook/bart-large-mnli"

_nlp_pipeline = None

def _load_pipeline():
    try:
        from transformers import pipeline
        logger.info(f"Chargement du modèle NLP : {PIPELINE_MODEL}")
        return pipeline("zero-shot-classification", model=PIPELINE_MODEL)
    except Exception as e:
        logger.warning(f"Modèle NLP non disponible ({e}), fallback rule-based")
        return None

def _get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _load_pipeline()
    return _nlp_pipeline

# ─── Rule-based fallback ────────────────────────────────────────────────

RULE_PATTERNS = {
    "diet": {"végétarien": r"\bvégétar\w*\b", "vegan": r"\bvegan\b", "keto": r"\bkéto\b|\bketo\b"},
    "goal": {"perdre du poids": r"\bperd\w*\b.*\bpoids\b|\bmaigr\w*\b"},
    "cuisine": {"asiatique": r"\basiat\w*\b"},
    "restriction": {"sans lactose": r"sans\s+lactose"},
    "protein": {"poulet": r"\bpoulet\b|\bvolaille\b"}
}

def _rule_based_extract(text: str) -> dict:
    text_lower = text.lower()
    results = {}
    for category, patterns in RULE_PATTERNS.items():
        found = []
        for label, pattern in patterns.items():
            if re.search(pattern, text_lower):
                found.append((label, 0.85))
        results[category] = found
    return results

# ─── NLP zero-shot ─────────────────────────────────────────────────────

def _classify(pipe, text: str, labels: List[str], multi: bool = True) -> List[tuple]:
    result = pipe(text, candidate_labels=labels, multi_label=multi)
    threshold = 0.35
    return [(label, round(score, 3)) for label, score in zip(result["labels"], result["scores"]) if score >= threshold]

def _nlp_extract(pipe, text: str) -> dict:
    return {
        "diet": _classify(pipe, text, DIET_LABELS, multi=False),
        "goal": _classify(pipe, text, GOAL_LABELS, multi=False),
        "cuisine": _classify(pipe, text, CUISINE_LABELS, multi=True),
        "restriction": _classify(pipe, text, RESTRICTION_LABELS, multi=True),
        "protein": _classify(pipe, text, PROTEIN_LABELS, multi=True),
    }

# ─── Service public ───────────────────────────────────────────────────

def analyze_preferences(text: str) -> dict:
    pipe = _get_pipeline()
    if pipe:
        raw = _nlp_extract(pipe, text)
        model = f"zero-shot ({PIPELINE_MODEL})"
    else:
        raw = _rule_based_extract(text)
        model = "rule-based (fallback)"

    def top(items, n=1) -> List[str]:
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return [label for label, _ in sorted_items[:n]] if sorted_items else []

    def scores(items) -> dict:
        return {label: score for label, score in items}

    diet_result = top(raw.get("diet", []))
    goal_result = top(raw.get("goal", []))
    proteins = [l for l, _ in raw.get("protein", [])]
    if not proteins:
        if "végétarien" in diet_result or "vegan" in diet_result:
            proteins = ["légumineuses", "tofu", "œufs"]
        else:
            proteins = ["poulet", "poisson"]

    return {
        "diet_type": diet_result[0] if diet_result else "omnivore",
        "goal": goal_result[0] if goal_result else "maintenir son poids",
        "cuisine_style": top(raw.get("cuisine", []), n=3),
        "restrictions": [l for l, _ in raw.get("restriction", [])],
        "preferred_proteins": proteins[:4],
        "confidence_scores": {
            "diet": scores(raw.get("diet", [])),
            "goal": scores(raw.get("goal", [])),
            "cuisine": scores(raw.get("cuisine", [])),
            "restriction": scores(raw.get("restriction", [])),
        },
        "model_used": model,
    }

# ─── FastAPI router ─────────────────────────────────────────────────────

router = APIRouter()

class PreferenceRequest(BaseModel):
    text: str

@router.post("/preferences/analyze")
def analyze_pref(req: PreferenceRequest):
    return analyze_preferences(req.text)