"""
COUCHE 2 — IA NLP : Analyse des préférences utilisateur
=========================================================
Modèle : facebook/bart-large-mnli (zero-shot classification)
"""

import logging
import re
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LABELS
# ─────────────────────────────────────────────

DIET_LABELS = [
    "végétarien", "vegan", "omnivore", "keto", "méditerranéen",
    "paléo", "sans gluten", "halal", "kasher", "flexitarien"
]

GOAL_LABELS = [
    "perdre du poids", "prendre de la masse musculaire",
    "maintenir son poids", "améliorer sa santé",
    "booster son énergie", "contrôler son diabète",
    "réduire le cholestérol"
]

CUISINE_LABELS = [
    "cuisine française", "cuisine asiatique", "cuisine méditerranéenne",
    "cuisine italienne", "cuisine mexicaine", "cuisine indienne",
    "cuisine japonaise", "cuisine africaine", "cuisine américaine"
]

RESTRICTION_LABELS = [
    "sans lactose", "sans gluten", "sans noix",
    "sans fruits de mer", "sans porc", "sans alcool",
    "faible en sodium", "faible en sucre"
]

PROTEIN_LABELS = [
    "poulet", "bœuf", "poisson", "légumineuses", "tofu",
    "œufs", "fruits de mer", "dinde", "agneau"
]

PIPELINE_MODEL = "facebook/bart-large-mnli"


# ─────────────────────────────────────────────
# PIPELINE LAZY LOADING
# ─────────────────────────────────────────────

_nlp_pipeline = None


def _load_pipeline():
    try:
        from transformers import pipeline
        logger.info(f"Loading NLP model: {PIPELINE_MODEL}")
        return pipeline(
            "zero-shot-classification",
            model=PIPELINE_MODEL
        )
    except Exception as e:
        logger.warning(f"NLP model not available → fallback rule-based ({e})")
        return None


def _get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _load_pipeline()
    return _nlp_pipeline


# ─────────────────────────────────────────────
# RULE-BASED FALLBACK
# ─────────────────────────────────────────────

RULE_PATTERNS = {
    "diet": {
        "végétarien": r"\bvégétar\w*\b",
        "vegan": r"\bvegan\b",
        "keto": r"\bkéto\b|\bketo\b",
        "méditerranéen": r"\bméditerra\w*\b",
        "paléo": r"\bpaléo\b|\bpaleo\b",
        "sans gluten": r"sans\s+gluten",
        "halal": r"\bhalal\b",
    },
    "goal": {
        "perdre du poids": r"\bperd\w*\b.*\bpoids\b|\bmaigr\w*\b",
        "prendre de la masse musculaire": r"\bmasse\b|\bmusc\w*\b",
        "maintenir son poids": r"\bmainti\w*\b|\bstabil\w*\b",
        "booster son énergie": r"\bénergi\w*\b",
        "améliorer sa santé": r"\bsant\w*\b",
    },
    "cuisine": {
        "asiatique": r"\basiat\w*\b|\bjaponais\w*\b|\bthaï\b",
        "méditerranéenne": r"\bméditerra\w*\b",
        "italienne": r"\bitalien\w*\b|\bpasta\b",
        "française": r"\bfrança\w*\b",
        "indienne": r"\bindien\w*\b|\bcurry\b",
        "mexicaine": r"\bmexicain\w*\b|\btaco\b",
    },
    "restriction": {
        "sans lactose": r"sans\s+lactose",
        "sans gluten": r"sans\s+gluten",
        "sans noix": r"sans\s+noix",
        "faible en sucre": r"sans\s+sucre|\bdiabét\w*\b",
    },
    "protein": {
        "poulet": r"\bpoulet\b",
        "poisson": r"\bpoisson\b|\bsaumon\b",
        "légumineuses": r"\blégumineuses\b|\blentilles\b",
        "tofu": r"\btofu\b|\bsoja\b",
        "œufs": r"\bœuf\w*\b|\begg\w*\b",
        "bœuf": r"\bbœuf\b|\bviande\b",
    }
}


def _rule_based_extract(text: str) -> dict:
    text = text.lower()
    results = {}

    for category, patterns in RULE_PATTERNS.items():
        found = []
        for label, pattern in patterns.items():
            if re.search(pattern, text):
                found.append((label, 0.85))
        results[category] = found

    return results


# ─────────────────────────────────────────────
# ZERO-SHOT NLP
# ─────────────────────────────────────────────

def _classify(pipe, text: str, labels: List[str], multi: bool = True):
    try:
        result = pipe(text, candidate_labels=labels, multi_label=multi)
        return [
            (label, float(score))
            for label, score in zip(result["labels"], result["scores"])
        ]
    except Exception as e:
        logger.warning(f"Classification error: {e}")
        return []


def _nlp_extract(pipe, text: str) -> dict:
    return {
        "diet": _classify(pipe, text, DIET_LABELS, multi=False),
        "goal": _classify(pipe, text, GOAL_LABELS, multi=False),
        "cuisine": _classify(pipe, text, CUISINE_LABELS),
        "restriction": _classify(pipe, text, RESTRICTION_LABELS),
        "protein": _classify(pipe, text, PROTEIN_LABELS),
    }


# ─────────────────────────────────────────────
# PUBLIC SERVICE
# ─────────────────────────────────────────────

def analyze_preferences(text: str) -> dict:
    pipe = _get_pipeline()

    if pipe:
        raw = _nlp_extract(pipe, text)
        model_used = f"zero-shot ({PIPELINE_MODEL})"
    else:
        raw = _rule_based_extract(text)
        model_used = "rule-based fallback"

    def top(items, n=1):
        return [
            label for label, _ in sorted(items, key=lambda x: x[1], reverse=True)[:n]
        ] if items else []

    def scores(items):
        return {label: float(score) for label, score in items}

    diet = top(raw.get("diet", []))
    goal = top(raw.get("goal", []))

    proteins = [l for l, _ in raw.get("protein", [])]

    if not proteins:
        if any(d in ["végétarien", "vegan"] for d in diet):
            proteins = ["légumineuses", "tofu", "œufs"]
        else:
            proteins = ["poulet", "poisson"]

    return {
        "diet_type": diet[0] if diet else "omnivore",
        "goal": goal[0] if goal else "maintenir son poids",
        "cuisine_style": top(raw.get("cuisine", []), 3),
        "restrictions": [l for l, _ in raw.get("restriction", [])],
        "preferred_proteins": proteins[:4],
        "confidence_scores": {
            "diet": scores(raw.get("diet", [])),
            "goal": scores(raw.get("goal", [])),
            "cuisine": scores(raw.get("cuisine", [])),
            "restriction": scores(raw.get("restriction", [])),
        },
        "model_used": model_used,
    }