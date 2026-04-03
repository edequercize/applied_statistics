FROM python:3.11-slim

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY models/ models/ 

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
