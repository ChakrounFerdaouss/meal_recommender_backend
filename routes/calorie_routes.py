from fastapi import APIRouter, HTTPException

from models.schemas import UserPhysicalData, CalorieEstimation
from services.calorie_dl import service as calorie_service

router = APIRouter()


# ─────────────────────────────────────────────
#  POST /calories/estimate (Deep Learning)
# ─────────────────────────────────────────────
@router.post("/calories/estimate", response_model=CalorieEstimation)
def estimate_user_calories(data: UserPhysicalData):
    """
    Couche 1 — IA Deep Learning

    Estime les besoins caloriques journaliers à partir d'un réseau de neurones
    entraîné sur dataset Hugging Face.
    """
    try:
        calories = calorie_service.predict(
            age=data.age,
            gender=1 if data.gender.value == "male" else 0,
            weight=data.weight,
            height=data.height,
            activity=_activity_to_float(data.activity.value)
        )

        bmi = data.weight / ((data.height / 100) ** 2)

        result = {
            "bmr": round(calories * 0.75, 1),
            "tdee": round(calories, 1),
            "bmi": round(bmi, 2),
            "bmi_category": _bmi_category(bmi),
            "model_used": "Deep Learning (CalorieNet)",
            "dataset_source": "khalidalt/DietNation"
        }

        return CalorieEstimation(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
#  GET /calories/model-info
# ─────────────────────────────────────────────
@router.get("/calories/model-info")
def calorie_model_info():
    """
    Informations sur le modèle Deep Learning utilisé
    """
    return {
        "model": "CalorieNet (PyTorch Neural Network)",
        "architecture": "6 features → 128 → 64 → 1",
        "dataset": "khalidalt/DietNation",
        "type": "Deep Learning Regression"
    }


# ─────────────────────────────────────────────
#  Helpers
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


def _bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Insuffisance pondérale"
    elif bmi < 25:
        return "Poids normal"
    elif bmi < 30:
        return "Surpoids"
    elif bmi < 35:
        return "Obésité modérée"
    else:
        return "Obésité sévère"
    
@router.get("/calories/debug")
def debug():
    return calorie_service.evaluate_model()