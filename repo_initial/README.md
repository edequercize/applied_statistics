# État initial du projet

Ce dossier contient l'état du repo tel qu'il était au départ, avant toute mise en production.

À ce stade, le projet se résumait à un seul notebook `main.ipynb` : exploration des données, preprocessing et entraînement de plusieurs modèles de classification.

Nous avons décidé de ne conserver qu'un seul modèle (LightGBM) et de suivre le parcours MLOps complet : bonnes pratiques de développement, fine-tuning reproductible via MLFlow, exposition via une API FastAPI, conteneurisation Docker, déploiement sur le SSP Cloud avec Kubernetes et ArgoCD, et monitoring en production.
