# ──────────────────────────────────────────────────────────────────────────────
# FastCrowdVision — Dockerfile
# Build: docker build -t fastcrowdvision:latest .
# Run:   docker run -p 8000:8000 fastcrowdvision:latest
# ──────────────────────────────────────────────────────────────────────────────

# --- Stage 1 : builder (installe les dépendances) ----------------------------
FROM python:3.11-slim AS builder

# Variables pour éviter les .pyc et le buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dépendances système pour OpenCV et torch (libGL, libgthread...)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copie uniquement le fichier de dépendances en premier (layer cache)
COPY requirements.txt .

# Installation des dépendances Python dans un venv dédié
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Stage 2 : image finale (allégée) ----------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Répertoire de cache HuggingFace — monté via PVC en prod
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Libs runtime seulement (pas de git/build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Récupération du venv construit à l'étape précédente
COPY --from=builder /opt/venv /opt/venv

# Copie du code applicatif
COPY . .

# Création d'un utilisateur non-root (bonne pratique sécurité)
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appuser /app

USER appuser

# Port exposé par uvicorn
EXPOSE 8000

# Healthcheck basique
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Démarrage de l'API avec uvicorn
CMD ["uvicorn", "server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "75"]
