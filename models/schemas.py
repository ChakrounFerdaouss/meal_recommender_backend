"""
Schémas Pydantic — validation et documentation automatique des données
"""

from pydantic import BaseModel, Field
from typing import List, Dict
from enum import Enum


# ─────────────────────────────────────────
# COUCHE 1 — IA Classique (Calories)
# ─────────────────────────────────────────

class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    active = "active"
    very_active = "very_active"


class Gender(str, Enum):
    male = "male"
    female = "female"


class UserPhysicalData(BaseModel):
    age: int = Field(..., ge=15, le=90)
    gender: Gender
    weight: float = Field(..., ge=30, le=300)
    height: float = Field(..., ge=100, le=250)
    activity: ActivityLevel


# ─────────────────────────────────────────
# MACROS + OBJECTIFS
# ─────────────────────────────────────────

class MacroNutrients(BaseModel):
    proteines_g: float
    lipides_g: float
    glucides_g: float


class CalorieGoals(BaseModel):
    perte_douce: int
    perte_rapide: int
    maintien: int
    prise_douce: int
    prise_rapide: int


class CalorieEstimation(BaseModel):
    tdee_kcal: float
    bmr_kcal: float
    bmi: float
    bmi_categorie: str
    activity_factor: float
    objectifs_caloriques: CalorieGoals
    macros_maintien: MacroNutrients


# ─────────────────────────────────────────
# COUCHE 2 — IA NLP (Préférences)
# ─────────────────────────────────────────

class PreferenceInput(BaseModel):
    text: str = Field(..., min_length=5)


class ConfidenceScores(BaseModel):
    diet: Dict[str, float]
    goal: Dict[str, float]
    cuisine: Dict[str, float]
    restriction: Dict[str, float]


class ExtractedPreferences(BaseModel):
    diet_type: str
    goal: str
    cuisine_style: List[str]
    restrictions: List[str]
    preferred_proteins: List[str]
    confidence_scores: ConfidenceScores
    model_used: str


# ─────────────────────────────────────────
# COUCHE 3 — IA GÉNÉRATIVE
# ─────────────────────────────────────────

class Mood(str, Enum):
    joyeux = "joyeux"
    stressé = "stressé"
    fatigué = "fatigué"
    motivé = "motivé"
    anxieux = "anxieux"
    serein = "serein"
    déprimé = "déprimé"


class EnergyLevel(str, Enum):
    très_faible = "très faible"
    faible = "faible"
    moyen = "moyen"
    élevé = "élevé"
    très_élevé = "très élevé"


class RecommendationRequest(BaseModel):
    calorie_data: CalorieEstimation
    preferences: ExtractedPreferences
    mood: Mood = Mood.motivé
    energy_level: EnergyLevel = EnergyLevel.moyen
    meals_per_day: int = Field(default=3, ge=2, le=6)
    days: int = Field(default=1, ge=1, le=7)


class Meal(BaseModel):
    name: str
    meal_type: str
    calories: int
    proteins_g: float
    carbs_g: float
    fats_g: float
    ingredients: List[str]
    instructions: str
    motivation: str


class DayPlan(BaseModel):
    day: int
    total_calories: int
    meals: List[Meal]


class MealRecommendation(BaseModel):
    plan: List[DayPlan]
    weekly_tips: List[str]
    hydration_advice: str
    model_used: str


# ─────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────

class FullPipelineRequest(BaseModel):
    physical_data: UserPhysicalData
    preference_text: str
    mood: Mood = Mood.motivé
    energy_level: EnergyLevel = EnergyLevel.moyen
    meals_per_day: int = Field(default=3, ge=2, le=6)
    days: int = Field(default=1, ge=1, le=3)


class FullPipelineResponse(BaseModel):
    step1_calories: CalorieEstimation
    step2_preferences: ExtractedPreferences
    step3_recommendations: MealRecommendation