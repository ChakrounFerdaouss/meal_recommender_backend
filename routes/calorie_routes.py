from fastapi import APIRouter, HTTPException

from models.schemas import UserPhysicalData, CalorieEstimation
from services.calorie_dl.service import service as calorie_service

router = APIRouter()


# ─────────────────────────────────────────────
# POST /calories/estimate
# ─────────────────────────────────────────────
@router.post("/calories/estimate", response_model=CalorieEstimation)
def estimate_user_calories(data: UserPhysicalData):
    """
    Couche 1 — IA Deep Learning

    Estime les besoins caloriques journaliers à partir du pipeline ML.
    """
    try:
        result = calorie_service.estimate(
            age=data.age,
            gender=data.gender.value,
            weight=data.weight,
            height=data.height,
            activity=data.activity.value,
        )

        return CalorieEstimation(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# GET /calories/model-info
# ─────────────────────────────────────────────
@router.get("/calories/model-info")
def calorie_model_info():
    """
    Informations sur le modèle Deep Learning utilisé.
    """
    try:
        return calorie_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# GET /calories/eval-report
# ─────────────────────────────────────────────
@router.get("/calories/eval-report")
def calorie_eval_report():
    """
    Retourne les métriques de performance du modèle.
    """
    try:
        return calorie_service.get_eval_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# GET /calories/tests
# ─────────────────────────────────────────────
@router.get("/calories/tests")
def run_calorie_tests():
    """
    Lance les tests sur profils de référence.
    """
    try:
        return calorie_service.run_tests()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POST /calories/retrain
# ─────────────────────────────────────────────
@router.post("/calories/retrain")
def retrain_model():
    """
    Relance l'entraînement complet du modèle.
    """
    try:
        return calorie_service.retrain()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))