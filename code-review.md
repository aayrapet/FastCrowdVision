# Code review

Cette PR constitue ma revue de code de votre projet.

---

## Appréciation générale

FastCrowdVision est un projet ambitieux et très bien réalisé. L'ensemble de la chaîne MLOps est couverte de bout en bout : entraînement, tracking, API, containerisation, déploiement Kubernetes et GitOps.

Quelques points qui méritent d'être soulignés :

- **README** très carré — instructions d'installation complètes, exemples d'appels API, section training, CI/CD.
- **Dockerfile multi-stage** avec utilisateur non-root, ce qui est généralement une très bonne practice. 
- Utilisation de `uv` pour l'installation est un bon choix, permet de résoudre les conflits de version.
- **API WebSocket** pour la détection en streaming est une solution élégante.
- La **séparation `model/` / `training/` / `serving/`** est claire et bien tenue, classique pour un projet MLOps propre.

---

## Conformité à la checklist

| Critère | Statut | Commentaire |
|---|---|---|
| Git & versioning | ✅ | Branches + PRs présentes |
| README + LICENSE | ✅ | README, MIT License présente |
| Gestion des dépendances | ❌ | Versions non épinglées (voir détail) |
| Qualité du code | ⚠️ | Ruff configuré mais absent de la CI |
| Structure du projet | ✅ | Séparation code/données/config respectée |
| Modèle ML | ⚠️ | Bon modèle, mais pas de fine-tuning formel des hyperparamètres |
| Tracking (MLflow) | ❌ | WandB utilisé à la place de MLflow |
| API FastAPI | ✅ | Fonctionnelle et déployée |
| Docker | ✅ | Multi-stage, non-root, dockerignore — très bien fait overall |
| Déploiement SSP Cloud | ✅ | Accessible publiquement |
| GitOps / ArgoCD / CI-CD | ⚠️ | ArgoCD présent ; CI sans tests ni lint |
| Monitoring | ❌ | Logging basique uniquement |

---

## Recommandations 

**1. Versions des dépendances non épinglées**

📄 `requirements/requirements.txt` — lignes 1-21
📄 `requirements/requirements-api.txt` — lignes 3-21

Dans `requirements/requirements.txt` et `requirements-api.txt`, la quasi-totalité des packages sont listés sans version (`torch`, `fastapi`, `uvicorn`, `pillow`, `huggingface_hub`…). Seul `numpy` a une contrainte. Sans versions fixes, une réinstallation dans quelques mois peut donner des résultats différents voire casser le proje - particulièrement risqué pour `torch` et `torchvision` qui ont des APIs qui évoluent.

Suggestion : épingler toutes les versions (`pip freeze > requirements-locked.txt`) ou utiliser `uv lock` pour générer un lockfile.

---

**2. Ruff et pytest absents de la CI**

📄 `.github/workflows/docker-deploy.yml` — après la ligne 22 (`actions/checkout@v4`)

`ruff` est correctement configuré dans `pyproject.toml`, et des tests existent dans `tests/` — c'est très bien. Mais ni l'un ni l'autre ne sont exécutés dans `.github/workflows/docker-deploy.yml`. Un code non conforme ou cassé peut donc être mergé et buildé sans alerte.

Suggestion : ajouter avant le step de build Docker :

```yaml
- name: Lint (ruff)
  run: pip install ruff && ruff check .

- name: Tests (pytest)
  run: pip install -r requirements/requirements.txt && pytest tests/ -v
```

---

**3. WandB utilisé à la place de MLflow**

📄 `training/train.py` — lignes 84-118

Le tracking des métriques avec WandB est bien fait (logs loss/mAP, reprise d'entraînement via `wandb_run_id`). Cependant, le cours demande explicitement l'utilisation de **MLflow**. Je n'ai pas l'impression que WandB joue pas ce rôle de la même façon.

Suggestion : conserver WandB pour la visualisation mais ajouter MLflow pour le tracking des runs et l'enregistrement du modèle dans un registry.

---

**4. Branche cible d'ArgoCD**

Dans `argocd/application.yaml`, `targetRevision` pointe sur `feat/argocd-migration` au lieu de `main`. Cela signifie qu'ArgoCD suit une branche de feature plutôt que la branche principale, ce qui peut créer une dérive entre prod et `main`.

---

**5. Pas de limite de taille sur l'upload vidéo**

Dans `server.py`, le contenu du fichier est chargé intégralement en mémoire :

```python
content = await file.read()
```

Sans limite de taille, une vidéo très lourde peut saturer la mémoire de Kubernetes. 

Suggestion : ajouter une limite de taille.

```python
MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB

content = await file.read(MAX_UPLOAD_SIZE + 1)
if len(content) > MAX_UPLOAD_SIZE:
    raise HTTPException(status_code=413, detail="Fichier trop volumineux (max 200 MB)")
```

---

### Moins important :

- **Déduplication** du bloc `wandb.init(...)` dans `train.py` : les branches `if multigpu / else` sont quasi-identiques — extraire en une fonction éviterait la répétition
- **Dashboard de monitoring** en production pour suivre la latence et le nombre de détections par frame par ex.

---

Overall : le projet est fonctionnel, déployé, et bien documenté, ce qui est l'essentiel !! 
