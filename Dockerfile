

FROM python:3.11-slim AS builder

# Copier uv depuis son image officielle (plus rapide et fiable que pip install uv)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Dépendances système minimales pour OpenCV headless et torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# → Docker met en cache cette couche et ne réinstalle pas si le code change
COPY requirements/requirements-api.txt .

# Créer un venv et installer les dépendances avec pip
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt


# ── Stage 2 : image finale allégée ───────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Libs runtime uniquement (pas d'outils de build)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Récupérer le venv du stage builder
COPY --from=builder /opt/venv /opt/venv

# Copier le code applicatif
# .dockerignore exclut : données, notebooks, scripts d'entraînement, .git
COPY . .

# Utilisateur non-root (bonne pratique sécurité Kubernetes)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "serving.server:app", "--host", "0.0.0.0", "--port", "8000"]
