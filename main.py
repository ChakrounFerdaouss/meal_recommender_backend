"""
Meal Recommender Backend
========================
3 couches d'IA :
  1. IA Deep Learning  → Réseau de neurones pour estimation des calories
  2. IA NLP           → Extraction des préférences utilisateur (zero-shot classification)
  3. IA Générative    → Recommandations via API (Claude / OpenAI fallback)
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.calorie_routes import router as calorie_router
from routes.preference_routes import router as preference_router
from routes.recommendation_routes import router as recommendation_router


# ─────────────────────────────────────────────
#  APP INIT
# ─────────────────────────────────────────────
app = FastAPI(
    title="Meal Recommender API",
    description="Backend intelligent de recommandation de repas personnalisés basé sur Deep Learning + NLP + IA Générative",
    version="2.0.0"
)


# ─────────────────────────────────────────────
#  CORS CONFIG
# ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
app.include_router(
    calorie_router,
    prefix="/api/v1",
    tags=["1. IA Deep Learning — Calories"]
)

app.include_router(
    preference_router,
    prefix="/api/v1",
    tags=["2. IA NLP — Préférences"]
)

app.include_router(
    recommendation_router,
    prefix="/api/v1",
    tags=["3. IA Générative — Recommandations"]
)


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "Meal Recommender API opérationnelle (DL + NLP + GenAI)",
        "architecture": {
            "layer1": "Deep Learning (CalorieNet - PyTorch)",
            "layer2": "NLP (zero-shot classification)",
            "layer3": "Generative AI (Claude / fallback OpenAI)"
        },
        "endpoints": {
            "calories":        "POST /api/v1/calories/estimate",
            "preferences":     "POST /api/v1/preferences/analyze",
            "recommendations": "POST /api/v1/recommendations/generate",
            "full_pipeline":   "POST /api/v1/recommendations/full",
            "model_info":      "GET /api/v1/calories/model-info",
            "docs":            "/docs"
        }
    }