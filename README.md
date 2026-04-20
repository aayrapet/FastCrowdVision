# FastCrowdVision

> Real-time people detection and counting on video, powered by a lightweight SSD model (SSDLite + MobileNetV3) trained on the WiderPeople dataset.

## Context and objective

FastCrowdVision is an end-to-end MLOps project: training an object detector optimized for mobile/edge devices, serving it through a FastAPI + WebSocket API with a web interface, containerizing with Docker, and deploying on Kubernetes (SSP Cloud).

The model detects **3 classes** from the WiderPeople dataset: pedestrians, riders, partially-visible persons The SSDLite + MobileNetV3 architecture is optimized to run on CPU. 

The trained model is available on HuggingFace: [aayrapet/SsdFastCrowdVision](https://huggingface.co/aayrapet/SsdFastCrowdVision)

> **Important — SSP Cloud limitation:** The SSP Cloud reverse-proxy blocks WebSocket connections. Even if the connection works and the uploading is possible, since video detection relies entirely on WebSocket streaming, the detection **will not work** when accessed through `https://fastcrowdvision.lab.sspcloud.fr`. You must run it locally instead (see below).

---

## Project structure

```
FastCrowdVision/
├── .github/workflows/
│   └── docker-deploy.yml        # CI/CD pipeline: build + push Docker image
├── argocd/
│   └── application.yaml         # ArgoCD manifest for GitOps deployment
├── config/                      # YAML backbone configuration files for SSD
├── datasets/
│   ├── WiderPeople/             # Download scripts for WiderPeople (Kaggle / S3)
│   └── voc/                     # Download scripts for VOC2007 (Kaggle / S3)
├── kubernetes/                  # Kubernetes manifests (deployment, service, ingress, pvc)
├── model/                       # SSD architecture and components
│   ├── ssd.py                   #   SSD / SSDLite
│   ├── mobilenetv2.py           #   MobileNetV2 backbone
│   ├── mobilenetv3.py           #   MobileNetV3 backbone
│   ├── detection.py             #   NMS post-processing
│   ├── priorbox.py              #   Anchor boxes
│   ├── l2norm.py                #   L2 normalization
│   └── utils.py                 #   Utility functions (matching, decode, etc.)
├── training/                    # Training pipeline
│   ├── train.py                 #   Training loop
│   ├── eval.py                  #   mAP evaluation + model loading
│   ├── multiloss.py             #   Multi-task loss function
│   ├── dataloader.py            #   PyTorch DataLoader
│   ├── transforms.py            #   Image transforms (train / test)
│   ├── multigpusetup.py         #   DDP multi-GPU setup
│   └── SsdTrainingPipelineVOC2007.py  # CLI training script
├── serving/                     # Detection API (what the Docker image runs)
│   ├── server.py                #   FastAPI + WebSocket
│   ├── inference.py             #   Model loading and per-frame detection
│   └── draw_inference.py        #   Inference visualization
├── scripts/                     # Standalone CLI tools
│   └── SsdFastCrowdVision.py    #   Image inference + FPS benchmark
├── website/                     # Static frontend (HTML/CSS/JS) served by FastAPI
├── tests/                       # Unit tests (backbone, SSD forward, HNM)
├── requirements/
│   ├── requirements.txt         #   Full dependencies (training + dev)
│   └── requirements-api.txt     #   Minimal dependencies (API / inference only)
├── Dockerfile                   # Multi-stage image (builder + runtime slim)
├── .dockerignore
├── .env.example
├── pyproject.toml               # Linter configuration (ruff)
└── README.md
```

---


## Get video


Access the video and upload it to website you will do in next steps: 


https://minio.lab.sspcloud.fr/aayrapetyan/FastCrowdVision/datasets/20260416_121332.mp4


## Running the application locally

Because SSP Cloud blocks WebSocket traffic, you need to run FastCrowdVision on your own machine. There are two options:

### Option A — Install from source

#### Prerequisites

- Python 3.11+
- (Optional) CUDA GPU for training

```bash
git clone https://github.com/aayrapet/FastCrowdVision.git
cd FastCrowdVision
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements/requirements-api.txt
uvicorn serving.server:app --reload
```

Then open [http://localhost:8000](http://localhost:8000), upload a video and start detection.

### Option B — Run with Docker

You can either build the image yourself:

```bash
docker build -t fastcrowdvision .
docker run -p 8000:8000 fastcrowdvision
```

Or pull and run the pre-built image directly from GitHub Container Registry:

```bash
docker run -d \
  --name fastcrowdvision \
  -p 8000:8000 \
  -e HF_HOME=/app/.cache/huggingface \
  -v hf-cache:/app/.cache/huggingface \
  ghcr.io/josiepierr/fastcrowdvision:latest
```

The `-v hf-cache:...` volume caches the model weights locally so they are only downloaded once.

The API is then accessible at [http://localhost:8000](http://localhost:8000).

---

## API endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | HTTP | Check that the server and model are ready |
| `POST /upload` | HTTP | Upload a video, returns a `session_id` |
| `WS /ws/detect` | WebSocket | Frame-by-frame detection, results streamed as JSON |

### Example `/upload` call

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@my_video.mp4"
# Returns: {"session_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
```

The WebSocket client then sends the configuration:

```json
{
  "session_id": "xxxxxxxx-...",
  "score_thr": 0.25,
  "frame_skip": 2
}
```

And receives for each processed frame:

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

> **Performance tip:** Without a GPU, set `frame_skip=2` or `frame_skip=3` in the web interface to process every 3rd frame and speed up detection.

---

## Training

Training requires downloading the dataset first, then running the training pipeline.

### 1. Download the data

Data is stored on S3 (SSP Cloud) and on Kaggle — code/data separation following MLOps best practices.

```bash
# From S3 (SSP Cloud)
python datasets/WiderPeople/s3/download.py
python datasets/voc/s3/download.py

# Or from Kaggle
python datasets/WiderPeople/kaggle/first_download.py
python datasets/voc/kaggle/first_download.py
```

### 2. Configure WandB (metric tracking)

```bash
cp .env.example .env
```

Edit `.env` and fill in your WandB credentials:

```
WANDB_API_KEY=<your_wandb_key>
ENTITY=<your_wandb_entity>
PROJECT=<your_wandb_project>
```

### 3. Launch training

```bash
python training/SsdTrainingPipelineVOC2007.py \
    'train-images_dir' \
    'train-label_dir' \
    'val-images_dir' \
    'val-label_dir' \
    'mobilenetv3large' \
    nb_classes \
    'ssd_voc2007_mv3large_aug' \
    --optimizer "adam" \
    --N_epochs 160 \
    --lr_schedule_epochs 156 170
```



Metrics (loss, mAP) are tracked in WandB throughout training, you can also restart training from last epoch, see more in details **SsdTrainingPipelineVOC2007**

---

## CI/CD and deployment

### CI/CD pipeline

The `.github/workflows/docker-deploy.yml` workflow triggers automatically on every push (all branches):

1. Builds the Docker image
2. Pushes to Docker Hub with a tag matching the branch name
3. The `latest` tag is only applied on pushes to `main`

**Required GitHub Secrets:**

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token (hub.docker.com > Account Settings > Security) |

### Deployment on SSP Cloud

After each CI build, redeploy the pod from the SSP Cloud terminal:

```bash
kubectl rollout restart deployment/fastcrowdvision
kubectl rollout status deployment/fastcrowdvision
```

### GitOps with ArgoCD

The `argocd/application.yaml` file defines an ArgoCD application configured to watch the `kubernetes/` folder in this repo with automatic sync (prune + self-heal).

> **Note:** Access to the `argocd` namespace on the SSP Cloud cluster is restricted to platform admins. Continuous deployment is therefore handled manually via `kubectl rollout restart` after each CI build.

---

## Tests

```bash
pytest tests/
```

Tests cover: SSD forward pass, MobileNetV2/V3 backbones, and Hard Negative Mining.

---

## References

- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- [ENSAE course — Putting models into production](https://ensae-reproductibilite.github.io/slides)
