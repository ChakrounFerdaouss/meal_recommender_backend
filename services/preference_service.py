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

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ─── Labels par dimension ────────────────────────────────────────────────────

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


# ─── Chargement du pipeline NLP ─────────────────────────────────────────────

def _load_pipeline():
    """
    Charge le pipeline zero-shot (téléchargement automatique depuis HuggingFace).
    ~1.6 Go en mémoire — singleton pour éviter les rechargements.
    """
    try:
        from transformers import pipeline
        logger.info(f"Chargement du modèle NLP : {PIPELINE_MODEL}")
        return pipeline("zero-shot-classification", model=PIPELINE_MODEL)
    except Exception as e:
        logger.warning(f"Modèle NLP non disponible ({e}), bascule sur l'extracteur rule-based")
        return None

_nlp_pipeline = None  # Chargé à la première requête (lazy loading)


def _get_pipeline():
    global _nlp_pipeline
    if _nlp_pipeline is None:
        _nlp_pipeline = _load_pipeline()
    return _nlp_pipeline


# ─── Extraction rule-based (fallback sans GPU) ───────────────────────────────

RULE_PATTERNS = {
    "diet": {
        "végétarien":    r"\bvégétar\w*\b",
        "vegan":         r"\bvegan\b",
        "keto":          r"\bkéto\b|\bketo\b",
        "méditerranéen": r"\bméditerra\w*\b",
        "paléo":         r"\bpaléo\b|\bpaleo\b",
        "sans gluten":   r"sans\s+gluten",
        "halal":         r"\bhalal\b",
    },
    "goal": {
        "perdre du poids":              r"\bperd\w*\b.*\bpoids\b|\bmaigr\w*\b|\bperte de poids\b",
        "prendre de la masse musculaire": r"\bmusc\w*\b|\bmasse\b|\bprise de masse\b",
        "maintenir son poids":          r"\bmainti\w*\b|\bstabil\w*\b",
        "booster son énergie":          r"\bénergi\w*\b|\bvitalit\w*\b",
        "améliorer sa santé":           r"\bsant\w*\b|\bbien.être\b",
    },
    "cuisine": {
        "asiatique":      r"\basiat\w*\b|\bjaponais\w*\b|\bchinois\w*\b|\bthaï\b",
        "méditerranéenne":r"\bméditerra\w*\b|\bgrec\w*\b",
        "italienne":      r"\bitalien\w*\b|\bpasta\b|\bpizza\b",
        "française":      r"\bfrança\w*\b",
        "indienne":       r"\bindien\w*\b|\bcurry\b",
        "mexicaine":      r"\bmexicain\w*\b|\btaco\b",
    },
    "restriction": {
        "sans lactose": r"sans\s+lactose|\bintolér\w*\s+au\s+lait\b",
        "sans gluten":  r"sans\s+gluten|\bcœliaque\b",
        "sans noix":    r"sans\s+noix|\ballerg\w*\s+aux\s+noix\b",
        "faible en sucre": r"sans\s+sucre|\bdiabét\w*\b|\bfaible.en.sucre\b",
    },
    "protein": {
        "poulet":       r"\bpoulet\b|\bvolaille\b",
        "poisson":      r"\bpoisson\b|\bsaumon\b|\bthon\b",
        "légumineuses": r"\blégumineuses\b|\blentilles\b|\bpois\b|\bharicots\b",
        "tofu":         r"\btofu\b|\bsoja\b|\btempeh\b",
        "œufs":         r"\bœuf\w*\b|\begg\w*\b",
        "bœuf":         r"\bbœuf\b|\bviande rouge\b|\bbifteck\b",
    }
}

def _rule_based_extract(text: str) -> dict:
    """Extraction par regex quand le modèle NLP n'est pas disponible."""
    text_lower = text.lower()
    results = {}

    for category, patterns in RULE_PATTERNS.items():
        found = []
        for label, pattern in patterns.items():
            if re.search(pattern, text_lower):
                found.append((label, 0.85))   # confiance fixe
        results[category] = found

    return results


# ─── Extraction via zero-shot NLP ────────────────────────────────────────────

def _classify(pipe, text: str, labels: List[str], multi: bool = True) -> List[Tuple[str, float]]:
    """Lance une classification zero-shot et retourne les labels > seuil."""
    result = pipe(text, candidate_labels=labels, multi_label=multi)
    threshold = 0.35
    return [
        (label, round(score, 3))
        for label, score in zip(result["labels"], result["scores"])
        if score >= threshold
    ]


def _nlp_extract(pipe, text: str) -> dict:
    """Extraction via transformers zero-shot sur toutes les dimensions."""
    return {
        "diet":        _classify(pipe, text, DIET_LABELS,        multi=False),
        "goal":        _classify(pipe, text, GOAL_LABELS,         multi=False),
        "cuisine":     _classify(pipe, text, CUISINE_LABELS,      multi=True),
        "restriction": _classify(pipe, text, RESTRICTION_LABELS,  multi=True),
        "protein":     _classify(pipe, text, PROTEIN_LABELS,      multi=True),
    }


# ─── Service public ─────────────────────────────────────────────────────────

def analyze_preferences(text: str) -> dict:
    """
    Point d'entrée principal de la couche 2.
    Retourne un profil structuré extrait du texte libre.
    """
    pipe = _get_pipeline()

    if pipe:
        raw   = _nlp_extract(pipe, text)
        model = f"zero-shot ({PIPELINE_MODEL})"
    else:
        raw   = _rule_based_extract(text)
        model = "rule-based (fallback)"

    def top(items, n=1) -> List[str]:
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        return [label for label, _ in sorted_items[:n]] if sorted_items else []

    def scores(items) -> dict:
        return {label: score for label, score in items}

    diet_result = top(raw.get("diet", []))
    goal_result = top(raw.get("goal", []))

    # Déduit les protéines préférées selon le régime si rien trouvé
    proteins = [l for l, _ in raw.get("protein", [])]
    if not proteins:
        if "végétarien" in diet_result or "vegan" in diet_result:
            proteins = ["légumineuses", "tofu", "œufs"]
        else:
            proteins = ["poulet", "poisson"]

    return {
        "diet_type":          diet_result[0] if diet_result else "omnivore",
        "goal":               goal_result[0] if goal_result else "maintenir son poids",
        "cuisine_style":      top(raw.get("cuisine", []), n=3),
        "restrictions":       [l for l, _ in raw.get("restriction", [])],
        "preferred_proteins": proteins[:4],
        "confidence_scores": {
            "diet":        scores(raw.get("diet", [])),
            "goal":        scores(raw.get("goal", [])),
            "cuisine":     scores(raw.get("cuisine", [])),
            "restriction": scores(raw.get("restriction", [])),
        },
        "model_used": model,
    }