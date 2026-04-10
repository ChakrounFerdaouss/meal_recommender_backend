from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Tuple, Optional
import logging
import re

logger = logging.getLogger(__name__)

router = APIRouter()

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
# PIPELINE
# ─────────────────────────────────────────────

_nlp_pipeline = None


def _load_pipeline():
    try:
        from transformers import pipeline
        logger.info(f"NLP model loading: {PIPELINE_MODEL}")
        return pipeline("zero-shot-classification", model=PIPELINE_MODEL)
    except Exception as e:
        logger.warning(f"NLP fallback activated: {e}")
        return None


def _get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _load_pipeline()
    return _nlp_pipeline


# ─────────────────────────────────────────────
# RULE BASED FALLBACK
# ─────────────────────────────────────────────

RULE_PATTERNS = {
    "diet": {
        "végétarien": r"\bvégétar\w*\b",
        "vegan": r"\bvegan\b",
        "keto": r"\bketo\b|\bkéto\b",
    },
    "goal": {
        "perdre du poids": r"\bperd\w*\b.*\bpoids\b|\bmaigr\w*\b",
        "musculation": r"\bmasse\b|\bmusc\w*\b",
    },
    "cuisine": {
        "asiatique": r"\basiat\w*\b",
    },
    "restriction": {
        "sans lactose": r"sans\s+lactose",
    },
    "protein": {
        "poulet": r"\bpoulet\b",
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
# NLP
# ─────────────────────────────────────────────

def _classify(pipe, text: str, labels: List[str], multi: bool = True) -> List[Tuple[str, float]]:
    try:
        result = pipe(text, candidate_labels=labels, multi_label=multi)
        return [
            (label, float(score))
            for label, score in zip(result["labels"], result["scores"])
            if score >= 0.35
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
# CORE SERVICE
# ─────────────────────────────────────────────

def analyze_preferences(text: str) -> dict:
    pipe = _get_pipeline()

    if pipe:
        raw = _nlp_extract(pipe, text)
        model_used = f"zero-shot ({PIPELINE_MODEL})"
    else:
        raw = _rule_based_extract(text)
        model_used = "rule-based"

    def top(items, n=1) -> List[str]:
        if not items:
            return []
        return [
            label for label, _ in sorted(items, key=lambda x: x[1], reverse=True)[:n]
        ]

    def scores(items) -> Dict[str, float]:
        return {label: float(score) for label, score in items} if items else {}

    diet = top(raw.get("diet", []))
    goal = top(raw.get("goal", []))

    proteins = [p for p, _ in raw.get("protein", [])]

    if not proteins:
        proteins = ["légumineuses", "tofu"] if "vegan" in diet else ["poulet", "poisson"]

    return {
        "diet_type": diet[0] if diet else "omnivore",
        "goal": goal[0] if goal else "maintenir son poids",
        "cuisine_style": top(raw.get("cuisine", []), 3),
        "restrictions": [r for r, _ in raw.get("restriction", [])],
        "preferred_proteins": proteins[:4],
        "confidence_scores": {
            "diet": scores(raw.get("diet", [])),
            "goal": scores(raw.get("goal", [])),
            "cuisine": scores(raw.get("cuisine", [])),
            "restriction": scores(raw.get("restriction", [])),
        },
        "model_used": model_used,
    }


# ─────────────────────────────────────────────
# FASTAPI ROUTE
# ─────────────────────────────────────────────

class PreferenceRequest(BaseModel):
    text: str


@router.post("/preferences/analyze")
def analyze_pref(req: PreferenceRequest):
    return analyze_preferences(req.text)