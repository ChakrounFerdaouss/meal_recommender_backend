"""
CalorieDLService — orchestre les 7 étapes du pipeline ML.

Usage API (FastAPI) :
    from services.calorie_dl import service
    result = service.estimate(age=28, gender="female", weight=62, height=166, activity="moderate")

Usage standalone (debug) :
    python -m services.calorie_dl.service
"""

import threading
import json
import os
import numpy as np

from .dataset     import collect_data
from .preprocessing import preprocess
from .features    import build_features
from .trainer     import CalorieTrainer
from .evaluator   import evaluate
from .inference   import predict_one, resolve_gender, resolve_activity, run_sample_tests
from .config      import ACTIVITY_MAP, REPORT_PATH


class CalorieDLService:
    def __init__(self):
        self.trainer     = CalorieTrainer()
        self._lock       = threading.Lock()
        self._source     = "unknown"
        self._eval_cache : dict | None = None

    # ── API publique ──────────────────────────────────────────────────────────

    def estimate(self, age: int, gender, weight: float,
                 height: float, activity) -> dict:
        """
        Estime les besoins caloriques journaliers.

        Returns:
            dict {tdee_kcal, bmr_kcal, bmi, bmi_categorie,
                  activity_factor, objectifs_caloriques, macros_maintien}
        """
        self._ensure_trained()

        g  = resolve_gender(gender)
        af = resolve_activity(activity)

        tdee = predict_one(
            self.trainer.model, self.trainer.scaler,
            age=age, gender=g, weight=weight,
            height=height, activity=af,
        )

        return self._build_report(age, g, weight, height, af, tdee)

    def get_model_info(self) -> dict:
        return {
            "trained":      self.trainer.trained,
            "architecture": "CalorieResNet (Residual MLP)",
            "parameters":   self.trainer.model.count_parameters(),
            "n_features":   8,
            "features":     ["age", "gender", "weight_kg", "height_cm",
                             "activity_factor", "bmr", "bmi", "weight_height_ratio"],
            "dataset":      self._source,
            "target":       "TDEE (kcal/jour)",
            "activity_levels": list(ACTIVITY_MAP.keys()),
        }

    def get_eval_report(self) -> dict:
        """Retourne les métriques de la dernière évaluation."""
        if self._eval_cache:
            return self._eval_cache
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH) as f:
                return json.load(f)
        return {"error": "Modèle non encore évalué"}

    def run_tests(self) -> list[dict]:
        """Lance les tests sur 6 profils de référence (étape 7)."""
        self._ensure_trained()
        return run_sample_tests(self.trainer.model, self.trainer.scaler)

    def retrain(self) -> dict:
        """Supprime le modèle sauvegardé et relance l'entraînement complet."""
        from .config import MODEL_PATH, SCALER_PATH
        for p in (MODEL_PATH, SCALER_PATH):
            if os.path.exists(p):
                os.remove(p)
        self.trainer.trained = False
        return self._run_full_pipeline()

    # ── Pipeline interne ──────────────────────────────────────────────────────

    def _ensure_trained(self):
        if self.trainer.trained:
            return
        with self._lock:
            if self.trainer.trained:
                return
            self._run_full_pipeline()

    def _run_full_pipeline(self) -> dict:
        print("\n" + "▓" * 52)
        print("  PIPELINE ML — CalorieDL")
        print("▓" * 52)

        # Étape 1 — Collecte
        print("\n── Étape 1 : Collecte des données ──")
        df, self._source = collect_data()

        # Étape 2 — Prétraitement
        print("\n── Étape 2 : Prétraitements ──")
        df_clean = preprocess(df)

        # Étape 3 — Représentation
        print("\n── Étape 3 : Représentation (feature engineering) ──")
        X, y = build_features(df_clean)

        # Étapes 4+5 — Modélisation + Entraînement
        print("\n── Étapes 4+5 : Modélisation & Entraînement ──")
        train_report = self.trainer.fit(X, y)

        # Étape 6 — Évaluation
        print("\n── Étape 6 : Test et évaluation ──")
        X_test = np.array(train_report.pop("X_test"), dtype=np.float32)
        y_test = np.array(train_report.pop("y_test"), dtype=np.float32)
        eval_metrics = evaluate(self.trainer.model, self.trainer.scaler, X_test, y_test)
        self._eval_cache = eval_metrics

        # Étape 7 — Test sur nouvelles données
        print("\n── Étape 7 : Test sur nouvelles données ──")
        run_sample_tests(self.trainer.model, self.trainer.scaler)

        return {**train_report, **eval_metrics}

    # ── Rapport de prédiction ─────────────────────────────────────────────────

    @staticmethod
    def _build_report(age, gender, weight, height, activity_factor, tdee) -> dict:
        bmr = 10*weight + 6.25*height - 5*age + (5 if gender == 1 else -161)
        bmi = weight / (height / 100) ** 2

        bmi_cat = (
            "Insuffisance pondérale" if bmi < 18.5 else
            "Poids normal"           if bmi < 25.0 else
            "Surpoids"               if bmi < 30.0 else
            "Obésité modérée (I)"    if bmi < 35.0 else
            "Obésité sévère (II)"    if bmi < 40.0 else
            "Obésité morbide (III)"
        )

        goals = {
            "perte_douce":  round(tdee * 0.85),
            "perte_rapide": round(tdee * 0.75),
            "maintien":     round(tdee),
            "prise_douce":  round(tdee * 1.10),
            "prise_rapide": round(tdee * 1.20),
        }

        def macros(kcal):
            return {
                "proteines_g": round(kcal * 0.30 / 4),
                "lipides_g":   round(kcal * 0.30 / 9),
                "glucides_g":  round(kcal * 0.40 / 4),
            }

        return {
            "tdee_kcal":             round(tdee, 1),
            "bmr_kcal":              round(bmr,  1),
            "bmi":                   round(bmi,  2),
            "bmi_categorie":         bmi_cat,
            "activity_factor":       activity_factor,
            "objectifs_caloriques":  goals,
            "macros_maintien":       macros(round(tdee)),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
service = CalorieDLService()