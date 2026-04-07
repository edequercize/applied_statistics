FROM python:3.11-slim-bookworm
WORKDIR /app

# LightGBM runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ src/
COPY api/ api/
COPY configs/ configs/

# Port exposé
EXPOSE 8000

# Lancement de l'API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]