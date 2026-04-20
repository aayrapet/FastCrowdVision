# FastCrowdVision

> Détection et comptage de personnes en temps réel sur vidéo, à partir d'un modèle SSD léger (SSDLite + MobileNetV3) entraîné sur le dataset WiderPeople.

## Contexte et objectif

FastCrowdVision est un projet MLOps de bout en bout : entraînement d'un détecteur d'objets optimisé pour les appareils mobiles, exposition via une API FastAPI avec interface web, conteneurisation Docker et déploiement sur le SSP Cloud (Kubernetes).

Le modèle détecte **5 classes** issues du dataset WiderPeople : piétons, cyclistes, personnes partiellement visibles, régions ignorées et foules. L'architecture SSDLite + MobileNetV3 est optimisée pour tourner sur CPU.

Le modèle entraîné est disponible sur HuggingFace : [aayrapet/SsdFastCrowdVision](https://huggingface.co/aayrapet/SsdFastCrowdVision)

L'application est déployée et accessible à : **https://fastcrowdvision.lab.sspcloud.fr**

---

## Architecture du projet

```
FastCrowdVision/
├── .github/
│   └── workflows/
│       └── docker-deploy.yml    # Pipeline CI/CD : build + push image Docker
├── argocd/
│   └── application.yaml         # Manifest ArgoCD pour déploiement GitOps
├── config/                      # Fichiers YAML de configuration des backbones SSD
│   ├── ssdlite_mobilenetv2.yaml
│   ├── ssdlite_mobilenetv3large.yaml
│   ├── ssdlite_mobilenetv3small.yaml
│   └── ssdlite_vgg.yaml
├── datasets/
│   ├── WiderPeople/             # Scripts téléchargement WiderPeople (Kaggle / S3)
│   └── voc/                     # Scripts téléchargement VOC2007 (Kaggle / S3)
├── kubernetes/                  # Manifestes Kubernetes (deployment, service, ingress, pvc)
├── models/                      # Package Python (__init__.py)
├── tests/                       # Tests unitaires (backbone, SSD forward, HNM)
├── website/                     # Frontend statique (HTML/CSS/JS) servi par FastAPI
├── .dockerignore
├── .env.example                 # Variables d'environnement à copier dans .env
├── .gitignore
├── Dockerfile                   # Image multi-stage (builder + runtime slim)
├── LICENSE
├── pyproject.toml               # Configuration linter (ruff)
├── README.md
├── requirements.txt             # Dépendances complètes (entraînement + API)
├── requirements-api.txt         # Dépendances minimales (API / inférence uniquement)
├── server.py                    # API FastAPI + WebSocket de détection
├── inference.py                 # Chargement du modèle et détection par frame
├── train.py                     # Boucle d'entraînement
├── dataloader.py                # DataLoader PyTorch
├── ssd.py                       # Architecture SSD
├── mobilenetv2.py               # Backbone MobileNetV2
├── mobilenetv3.py               # Backbone MobileNetV3
├── multiloss.py                 # Fonction de perte multi-tâche
├── detection.py                 # Post-traitement des détections
├── eval.py                      # Évaluation mAP
├── transforms.py                # Transformations image (train / test)
├── utils.py                     # Fonctions utilitaires
└── draw_inference.py            # Visualisation des inférences
```

---

## Installation et lancement local

### Prérequis

- Python 3.11+
- (Optionnel) GPU CUDA pour l'entraînement

### 1. Cloner le dépôt

```bash
git clone https://github.com/aayrapet/FastCrowdVision.git
cd FastCrowdVision
```

### 2. Configurer l'environnement

```bash
python -m venv .venv
source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements/requirements.txt
```

### 3. Configurer les variables d'environnement pour le training

```bash
cp .env.example .env
```

Édite `.env` et renseigne tes clés WandB (que pour le training):

```
WANDB_API_KEY=<ta_clé_wandb>
ENTITY=<ton_entity_wandb>
PROJECT=<nom_du_projet_wandb>
```

### 4. Lancer l'API en local (en inférence)

```bash
uvicorn server:app --reload
```

Ouvre ensuite [http://localhost:8000](http://localhost:8000), uploade une vidéo et lance la détection.

---

## Utilisation de l'API

| Endpoint | Méthode | Description |
|---|---|---|
| `GET /health` | HTTP | Vérifie que le serveur et le modèle sont prêts |
| `POST /upload` | HTTP | Upload une vidéo, retourne un `session_id` |
| `WS /ws/detect` | WebSocket | Détection frame par frame, résultats en streaming JSON |

### Exemple d'appel `/upload`

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@ma_video.mp4"
# Retourne : {"session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

Le client WebSocket envoie ensuite la configuration :

```json
{
  "session_id": "xxxxxxxx-...",
  "score_thr": 0.25,
  "frame_skip": 2
}
```

Et reçoit pour chaque frame traitée :

```json
{
  "type": "detection",
  "frame": 42,
  "time": 1.4,
  "boxes": [[x1, y1, x2, y2]],
  "track_ids": [1, 3],
  "scores": [0.87, 0.72],
  "classes": ["pedestrians", "riders"],
  "current_count": 2,
  "total_unique": 5
}
```

> **Astuce performance :** Sans GPU, utilise `frame_skip=2` ou `frame_skip=3` dans l'interface pour traiter 1 frame sur 3 et accélérer la détection.

---

## Lancement avec Docker

```bash
# Build de l'image
docker build -t fastcrowdvision .

# Lancement
docker run -p 8000:8000 fastcrowdvision
```

L'API est accessible sur [http://localhost:8000](http://localhost:8000).

---

## Entraînement

Les données sont stockées sur S3 (SSP Cloud) et sur Kaggle — séparation code/données conforme aux bonnes pratiques MLOps.

```bash
# Depuis S3 (SSP Cloud)
python datasets/WiderPeople/s3/download.py
python datasets/voc/s3/download.py

# Ou depuis Kaggle
python datasets/WiderPeople/kaggle/first_download.py
```

Lancer l'entraînement :

```bash
python train.py
```

Le suivi des métriques (loss, mAP) est assuré par WandB. Assure-toi que `.env` est bien configuré.

---

## CI/CD et déploiement

### Pipeline CI/CD

Le fichier `.github/workflows/docker-deploy.yml` déclenche automatiquement sur chaque push (toutes branches) :

1. Build de l'image Docker
2. Push sur Docker Hub avec le tag correspondant à la branche
3. Le tag `latest` est réservé aux pushs sur `main`

**Secrets GitHub à configurer :**

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Identifiant Docker Hub |
| `DOCKERHUB_TOKEN` | Access Token Docker Hub (hub.docker.com → Account Settings → Security) |

### Déploiement sur SSP Cloud

Après chaque build CI, redéployer le pod depuis le terminal SSP Cloud :

```bash
kubectl rollout restart deployment/fastcrowdvision
kubectl rollout status deployment/fastcrowdvision
```

### GitOps avec ArgoCD

Le fichier `argocd/application.yaml` définit l'application ArgoCD configurée pour surveiller automatiquement le dossier `kubernetes/` du repo avec synchronisation automatique (prune + self-heal).

> **Note :** L'accès au namespace `argocd` du cluster SSP Cloud est restreint aux admins de la plateforme. Le déploiement continu est donc assuré manuellement via `kubectl rollout restart` après chaque build CI.

---

## Tests

```bash
pytest tests/
```

Les tests couvrent : le forward pass SSD, les backbones MobileNetV2/V3, et le Hard Negative Mining.

---

## Références

- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [Cours ENSAE — Mise en production](https://ensae-reproductibilite.github.io/slides)
