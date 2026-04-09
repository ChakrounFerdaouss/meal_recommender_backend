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

# 🔥 Deep Learning service
from services.calorie_dl import service as calorie_service


router = APIRouter()


# ─────────────────────────────────────────────
# POST /recommendations/generate
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

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POST /recommendations/text
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
        return {"plan_text": text, "model": "gpt-4o-mini"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POST /recommendations/full
# ─────────────────────────────────────────────
@router.post("/recommendations/full", response_model=FullPipelineResponse)
def full_pipeline(data: FullPipelineRequest):
    try:
        # ── Couche 1 : DEEP LEARNING ──
        calories_pred = calorie_service.predict(
            age=data.physical_data.age,
            gender=1 if data.physical_data.gender.value == "male" else 0,
            weight=data.physical_data.weight,
            height=data.physical_data.height,
            activity=_activity_to_float(data.physical_data.activity.value)
        )

        bmi = data.physical_data.weight / ((data.physical_data.height / 100) ** 2)

        calorie_raw = {
            "bmr": round(calories_pred * 0.75, 1),
            "tdee": round(calories_pred, 1),
            "bmi": round(bmi, 2),
            "bmi_category": "computed",
            "model_used": "CalorieNet (Deep Learning)",
            "dataset_source": "khalidalt/DietNation"
        }

        calorie_result = CalorieEstimation(**calorie_raw)

        # ── Couche 2 : NLP ──
        pref_raw = analyze_preferences(data.preference_text)
        pref_result = ExtractedPreferences(**pref_raw)

        # ── Couche 3 : GEN AI ──
        reco_raw = generate_recommendations(
            calorie_data=calorie_raw,
            preferences=pref_raw,
            mood=data.mood.value,
            energy_level=data.energy_level.value,
            meals_per_day=data.meals_per_day,
            days=data.days
        )

        reco_result = MealRecommendation(**reco_raw)

        return FullPipelineResponse(
            step1_calories=calorie_result,
            step2_preferences=pref_result,
            step3_recommendations=reco_result
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _activity_to_float(activity: str) -> float:
    mapping = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9,
    }
    return mapping.get(activity, 1.55)