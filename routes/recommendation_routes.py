from fastapi import APIRouter, HTTPException

from models.schemas import (
    RecommendationRequest,
    MealRecommendation,
    FullPipelineRequest,
    FullPipelineResponse,
    CalorieEstimation,
    ExtractedPreferences,
    Mood,
    EnergyLevel
)

from services.preference_service import analyze_preferences
from services.recommendation_service import (
    generate_recommendations,
    generate_meal_plan
)

from services.calorie_dl.service import service as calorie_service

router = APIRouter()


# ─────────────────────────────────────────────
# /recommendations/generate
# ─────────────────────────────────────────────

@router.post("/recommendations/generate", response_model=MealRecommendation)
def generate_meal_recommendations(data: RecommendationRequest):

    try:
        result = generate_recommendations(
            calorie_data=data.calorie_data.model_dump(),
            preferences=data.preferences.model_dump(),
            mood=data.mood.value,
            energy_level=data.energy_level.value,
            meals_per_day=data.meals_per_day,
            days=data.days
        )

        return MealRecommendation(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# /recommendations/text
# ─────────────────────────────────────────────

@router.post("/recommendations/text")
def generate_text_plan(
    calories: float,
    preferences: str,
    objectif: str,
    mood: Mood = Mood.motivé,
    energy_level: EnergyLevel = EnergyLevel.moyen
):

    try:
        text = generate_meal_plan(
            calories=calories,
            preferences=preferences,
            objectif=objectif,
            mood=mood.value,
            energy_level=energy_level.value
        )

        return {
            "plan_text": text,
            "model": "gpt-4o-mini"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# /recommendations/full (PIPELINE 3 COUCHES)
# ─────────────────────────────────────────────

@router.post("/recommendations/full", response_model=FullPipelineResponse)
def full_pipeline(data: FullPipelineRequest):

    try:
        # ─────────────────────────────
        # COUCHE 1 — DL CALORIES
        # ─────────────────────────────
        calorie_result_dict = calorie_service.estimate(
            age=data.physical_data.age,
            gender=data.physical_data.gender.value,
            weight=data.physical_data.weight,
            height=data.physical_data.height,
            activity=data.physical_data.activity.value
        )

        calorie_result = CalorieEstimation(**calorie_result_dict)

        # ─────────────────────────────
        # COUCHE 2 — NLP
        # ─────────────────────────────
        pref_dict = analyze_preferences(data.preference_text)
        pref_result = ExtractedPreferences(**pref_dict)

        # ─────────────────────────────
        # COUCHE 3 — GEN AI
        # ─────────────────────────────
        reco_dict = generate_recommendations(
            calorie_data=calorie_result_dict,
            preferences=pref_dict,
            mood=data.mood.value,
            energy_level=data.energy_level.value,
            meals_per_day=data.meals_per_day,
            days=data.days
        )

        reco_result = MealRecommendation(**reco_dict)

        return FullPipelineResponse(
            step1_calories=calorie_result,
            step2_preferences=pref_result,
            step3_recommendations=reco_result
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))