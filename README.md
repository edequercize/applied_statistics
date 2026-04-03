# 🏥 Obesity Risk Classification — MLOps Project

## Contexte

Ce projet vise à prédire le **niveau d'obésité** d'un individu à partir de ses habitudes alimentaires, de son activité physique et de caractéristiques démographiques. Les données proviennent du dataset [Obesity Levels (UCI)](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

Il s'agit d'un problème de **classification ordinale multi-classes** (7 niveaux) :
`Insufficient_Weight → Normal_Weight → Overweight I/II → Obesity I/II/III`

## Objectif

- Développer un modèle de ML performant pour classifier le niveau d'obésité
- Suivre les bonnes pratiques de développement (Git, modularité, qualité de code)
- Déployer le modèle en production via un parcours **MLOps** complet

## Structure du projet

```
applied_statistics/
├── src/                    # Code source modulaire
│   ├── data/
│   │   ├── load.py         # Chargement des données
│   │   └── preprocess.py   # Feature engineering & pipeline sklearn
│   ├── models/
│   │   ├── train.py        # Entraînement + CV + fine-tuning
│   │   └── evaluate.py     # Métriques (MAE ordinale, confusion matrix)
│   └── utils.py            # Helpers (mappings, encodages)
├── notebooks/
│   └── eda.ipynb           # Exploration des données (visualisations)
├── api/
│   ├── app.py              # API FastAPI
│   └── schemas.py          # Schémas Pydantic
├── tests/
│   └── test_preprocess.py  # Tests unitaires
├── configs/
│   └── params.yaml         # Hyperparamètres centralisés
├── main.py                 # Point d'entrée CLI
├── Dockerfile
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/edequercize/applied_statistics.git
cd applied_statistics
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Données

Placer le fichier CSV dans `data/` :
```
data/ObesityDataSet_raw_and_data_sinthetic.csv
```

> Les données ne sont pas versionnées (voir `.gitignore`).

## Utilisation

### Entraînement
```bash
python main.py --model lightgbm --config configs/params.yaml
```

### Lancer l'API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### Lancer les tests
```bash
pytest tests/
```

### Linting
```bash
ruff check src/ api/ tests/
ruff format src/ api/ tests/
```

## MLflow

```bash
mlflow ui --port 5000
```
Puis ouvrir http://localhost:5000 pour visualiser les runs.

## Docker

```bash
docker build -t obesity-api .
docker run -p 8000:8000 obesity-api
```

## Licence

MIT — voir [LICENSE](LICENSE).