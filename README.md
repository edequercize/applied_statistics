# Obesity Risk Classification — MLOps Project

## Contexte

Ce projet vise à prédire le **niveau d'obésité** d'un individu à partir de ses habitudes alimentaires, de son activité physique et de caractéristiques démographiques. Les données proviennent du dataset [Obesity Levels (UCI)](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).

Il s'agit d'un problème de **classification ordinale multi-classes** (7 niveaux) :

`Insufficient_Weight → Normal_Weight → Overweight_I → Overweight_II → Obesity_I → Obesity_II → Obesity_III`

## Architecture MLOps

```
Code (GitHub) → CI/CD (GitHub Actions) → Image Docker (Docker Hub)
                                                    ↓
                                         ArgoCD (GitOps) → Kubernetes (SSP Cloud)
                                                    ↓
                                         API FastAPI + Prometheus + Grafana
```

## Structure du projet

```
applied_statistics/
├── src/                        # Code source modulaire
│   ├── data/
│   │   ├── load.py             # Chargement données (CSV, Parquet, S3, URL publique)
│   │   └── preprocess.py       # Feature engineering & pipeline sklearn
│   ├── models/
│   │   ├── train.py            # Entraînement LightGBM + GridSearchCV
│   │   └── evaluate.py         # Métriques (MAE ordinale, matrice de confusion)
│   └── utils.py                # Helpers (mappings ordinaux, encodages)
├── api/
│   ├── app.py                  # API FastAPI (health, predict, metrics)
│   └── schemas.py              # Schémas Pydantic
├── tests/
│   └── test_preprocess.py      # Tests unitaires
├── configs/
│   └── params.yaml             # Hyperparamètres centralisés
├── deployment/                 # Manifestes Kubernetes
│   ├── deployment.yaml         # Déploiement de l'API
│   ├── service.yaml            # Service K8s
│   ├── ingress.yaml            # Ingress (routing HTTP/S)
│   └── argocd-application.yaml # Application ArgoCD (GitOps)
├── notebooks/
│   └── eda.ipynb               # Exploration des données
├── main.py                     # Point d'entrée CLI (entraînement)
├── Dockerfile
├── requirements.txt
├── .github/workflows/
│   └── prod.yml                # CI/CD : build & push image Docker
└── README.md
```

## Installation locale

```bash
git clone https://github.com/edequercize/applied_statistics.git
cd applied_statistics
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

## Données

Les données sont stockées sur **MinIO (SSP Cloud)** en format Parquet et accessibles publiquement :

```
https://minio.lab.sspcloud.fr/edq/data/ObesityDataSet_raw_and_data_sinthetic.parquet
```

Le chemin est configuré dans `configs/params.yaml`. Le chargement utilise **DuckDB** pour lire le Parquet efficacement (local, URL publique ou S3).

> Les données ne sont pas versionnées dans Git (voir `.gitignore`).

## Entraînement

```bash
python main.py
# ou avec un fichier de config spécifique :
python main.py --config configs/params.yaml
```

Le script :
1. Charge les données depuis MinIO (Parquet via DuckDB)
2. Prétraite les features (OHE, colonnes ordinales)
3. Lance un **GridSearchCV** LightGBM (cross-validation 3 folds)
4. Loggue les métriques et le modèle dans **MLflow**
5. Sauvegarde le meilleur modèle localement (`models/best_model.joblib`)
6. Uploade le modèle sur **MinIO S3** (`s3://edq/models/best_model.joblib`)

### Visualiser les runs MLflow

```bash
mlflow ui --port 5000
```

Puis ouvrir [http://localhost:5000](http://localhost:5000).

## API FastAPI

### Lancer localement

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Le modèle est téléchargé automatiquement depuis MinIO au démarrage si absent localement (variable `MODEL_URL`).

### Endpoints

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/health` | État de l'API et du modèle |
| POST | `/predict` | Prédiction du niveau d'obésité |
| GET | `/metrics` | Métriques Prometheus |
| GET | `/docs` | Documentation Swagger interactive |

### Exemple de prédiction

```bash
curl -X POST https://obesity-api-test.lab.sspcloud.fr/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 25,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 3,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 1,
    "TUE": 1,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
  }'
```

**API en production** : [https://obesity-api-test.lab.sspcloud.fr/docs](https://obesity-api-test.lab.sspcloud.fr/docs)

## Docker

### Build et run local

```bash
docker build -t obesity-api .
docker run -p 8000:8000 \
  -e MODEL_URL=https://minio.lab.sspcloud.fr/edq/models/best_model.joblib \
  obesity-api
```

Puis ouvrir [http://localhost:8000/docs](http://localhost:8000/docs).

### CI/CD

Le workflow `.github/workflows/prod.yml` déclenche automatiquement à chaque push sur `main` :
1. Build de l'image Docker
2. Push sur Docker Hub : `ashxivy/applied_statistics:latest` (+ tag SHA pour rollback)

## Déploiement SSP Cloud (GitOps)

Le déploiement est géré via **ArgoCD** sur le cluster Kubernetes de SSP Cloud (`user-edq`).

ArgoCD surveille le repo [Darren6414/application-deployment](https://github.com/Darren6414/application-deployment) et applique automatiquement les manifestes K8s à chaque push.

### Manifestes déployés

- **API** : `deployment.yaml` + `service.yaml` + `ingress.yaml`
- **Prometheus** : scrape les métriques de l'API toutes les 15s
- **Grafana** : dashboards de monitoring accessibles via ingress

### Modèle en production

Le modèle n'est pas embarqué dans l'image Docker. Au démarrage du pod, l'API le télécharge automatiquement depuis MinIO via URL publique HTTPS :

```
https://minio.lab.sspcloud.fr/edq/models/best_model.joblib
```

## Monitoring

| Outil | URL | Description |
|-------|-----|-------------|
| API Health | [/health](https://obesity-api-test.lab.sspcloud.fr/health) | État du modèle |
| Métriques Prometheus | [/metrics](https://obesity-api-test.lab.sspcloud.fr/metrics) | Latence, requêtes, mémoire |
| Grafana | [user-edq-grafana.user.lab.sspcloud.fr](https://user-edq-grafana.user.lab.sspcloud.fr) | Dashboards de performance |

## Tests et qualité de code

```bash
# Tests unitaires
pytest tests/

# Linting et formatage
ruff check src/ api/ tests/
ruff format src/ api/ tests/
```

## Licence

MIT — voir [LICENSE](LICENSE).
