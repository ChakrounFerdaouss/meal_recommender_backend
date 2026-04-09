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
│   ├── calorie_service.py           ← COUCHE 1 : IA Classique (Ridge + HuggingFace)
│   ├── preference_service.py        ← COUCHE 2 : IA NLP (zero-shot BART)
│   └── recommendation_service.py   ← COUCHE 3 : IA Générative (Claude API)
├── routes/
│   ├── calorie_routes.py
│   ├── preference_routes.py
│   └── recommendation_routes.py
└── tests/
    └── test_all.py
```

## Les 3 couches d'IA

| # | Type | Technologie | Rôle |
|---|------|------------|------|
| 1 | IA Classique | Ridge Regression + Dataset HuggingFace `khalidalt/DietNation` | Estimer BMR, TDEE, IMC |
| 2 | IA NLP | `facebook/bart-large-mnli` (zero-shot) | Extraire régime, objectif, restrictions du texte |
| 3 | IA Générative | Anthropic Claude `claude-sonnet-4-20250514` | Générer le plan de repas motivant |

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
    "weight": 75.0,
    "height": 178.0,
    "activity": "moderate"
  }'
```

#### Couche 2 — Analyser les préférences
```bash
curl -X POST http://localhost:8000/api/v1/preferences/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Je suis végétarienne, j adore la cuisine méditerranéenne et asiatique. Mon objectif est de perdre 5kg tout en gardant mon énergie."
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
      "weight": 62.0,
      "height": 166.0,
      "activity": "moderate"
    },
    "preference_text": "Je suis végétarienne, j adore la cuisine méditerranéenne et asiatique. Mon objectif est de perdre 5kg.",
    "meals_per_day": 3,
    "days": 1
  }'
```

---

## Endpoints disponibles

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Health check + liste des endpoints |
| GET | `/docs` | Documentation Swagger interactive |
| POST | `/api/v1/calories/estimate` | Couche 1 : estimation calorique |
| GET | `/api/v1/calories/model-info` | Infos modèle ML et dataset |
| POST | `/api/v1/preferences/analyze` | Couche 2 : analyse NLP du texte |
| POST | `/api/v1/recommendations/generate` | Couche 3 : génération Claude |
| **POST** | **`/api/v1/recommendations/full`** | **Pipeline complet (1+2+3)** |

---

## Notes importantes

- **Première exécution** : le dataset HuggingFace (~2 Mo) et le modèle BART (~1.6 Go) sont téléchargés automatiquement
- **Sans GPU** : remplacer `torch` par `torch-cpu` dans requirements.txt pour réduire la taille
- **Fallback NLP** : si `transformers` n'est pas disponible, l'extraction se fait par règles regex
- **Fallback dataset** : si HuggingFace est inaccessible, des données synthétiques sont utilisées