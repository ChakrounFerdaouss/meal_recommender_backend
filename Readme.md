# Meal Recommender — Backend IA

Backend intelligent de recommandation de repas personnalisés combinant **3 couches d'IA**.

## Architecture

```
meal-recommender/
├── main.py                          ← Point d'entrée FastAPI
├── requirements.txt
├── .env                             ← Variables d'environnement (à créer)
├── models/
│   └── schemas.py                   ← Schémas Pydantic (validation)
├── services/
│   ├── calorie_dl                 ← COUCHE 1 : IA Classique (Deeplearning)
│   ├── preference_service.py        ← COUCHE 2 : IA NLP (zero-shot BART)
│   └── recommendation_service.py   ← COUCHE 3 : IA Générative (OpenAI API)
├── routes/
│   ├── calorie_routes.py
│   ├── preference_routes.py
│   └── recommendation_routes.py
```

## Les 3 couches d'IA

| # | Type | Technologie | Rôle |
|---|------|------------|------|
| 1 | Deep Learning | PyTorch + dataset `khalidalt/DietNation` | Estimer BMR, TDEE, IMC |
| 2 | NLP | `facebook/bart-large-mnli` (zero-shot) | Extraire régime, objectifs et restrictions |
| 3 | Générative | OpenAI `gpt-4o-mini` (fallback possible) | Générer des plans de repas
---

## Installation

### 1. Cloner et préparer l'environnement

```bash
cd meal-recommender
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Configurer la clé API

```bash
# Créer le fichier .env
echo "OPENAI_API_KEY=sk-ant-xxxxxxxxxxxx" > .env
```

### 3. Lancer le serveur

```bash
uvicorn main:app --reload --port 8000
```

Le serveur démarre sur `http://localhost:8000`

---

## Tester l'API

### Documentation interactive (Swagger)
```
http://localhost:8000/docs
```

### Tests manuels avec curl

#### Couche 1 — Estimer les calories
```bash
curl -X POST http://localhost:8000/api/v1/calories/estimate \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "gender": "male",
    "weight": 75,
    "height": 178,
    "activity": "moderate"
  }'
```

#### Couche 2 — Analyser les préférences
```bash
curl -X POST http://localhost:8000/api/v1/preferences/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Je suis végétarienne, j aime la cuisine méditerranéenne et asiatique. Mon objectif est de perdre du poids."
  }'
```

#### Pipeline complet (recommandé)
```bash
curl -X POST http://localhost:8000/api/v1/recommendations/full \
  -H "Content-Type: application/json" \
  -d '{
    "physical_data": {
      "age": 28,
      "gender": "female",
      "weight": 62,
      "height": 166,
      "activity": "moderate"
    },
    "preference_text": "Je suis végétarienne, j aime la cuisine méditerranéenne et asiatique. Mon objectif est de perdre du poids.",
    "mood": "motivé",
    "energy_level": "moyen",
    "meals_per_day": 3,
    "days": 1
  }'
```

---

## Endpoints disponibles

Méthode	Endpoint	Description
GET	/	Health check
GET	/docs	Swagger UI
POST	/api/v1/calories/estimate	Estimation BMR/TDEE
GET	/api/v1/calories/model-info	Infos modèle DL
GET	/api/v1/calories/debug	Metrics du modèle
POST	/api/v1/preferences/analyze	Analyse NLP
POST	/api/v1/recommendations/generate	Génération IA
POST	/api/v1/recommendations/full	Pipeline complet

---

## Notes importantes

🧠 Modèle DL basé sur PyTorch (CalorieNet)
🤖 NLP avec facebook/bart-large-mnli
⚡ Génération avec OpenAI (gpt-4o-mini)
🔁 Fallback automatique si API indisponible
📦 Swagger disponible sur /docs