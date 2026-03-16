# FastCrowdVision — Onboarding Guide

## 1. Project Goal

Detect **people in video feeds** (CCTV cameras) in real time.

We have **two deployment targets**:

| Target | Description |
|---|---|
| **Minimum (most probable)** | Run inference **locally on a PC** through a JavaScript web application. The user opens a local site, feeds a video, and sees bounding boxes drawn on detected people. |
| **Maximum** | Run the model directly **on a mobile / edge device** (phone, embedded board). This is why we focus on lightweight model variants. |

The entire model is implemented from scratch in PyTorch so we have full control over what we ship.

---

## 2. How the Model Works 

The model takes a single image and, in **one forward pass**, outputs bounding boxes around every person it finds together with a confidence score. There are three stages:

```
Input Image (300 × 300)
        │
        ▼
┌───────────────────┐
│     Backbone      │   A pretrained neural network that looks at the image
│                   │   and extracts visual features (edges, shapes, body parts…).
│                   │   We support 4 backbones — from heavy to ultra-light.(vgg is heavy, mobilenetv3 is ultralite)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│    Detection      │   Takes those features and, for thousands of predefined
│                   │   candidate boxes spread across the image, predicts:
│                   │     1. Classification — "is there a person here? yes/no"
│                   │     2. Bounding box   — "adjust this box to fit the person"
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Post-process    │   Removes duplicate / low-confidence boxes and keeps
│                   │   only the best detections (Non-Max Suppression).
└───────────────────┘
         │
         ▼
  Final output: list of bounding boxes + confidence scores
```

**In short: Backbone → Detection (classification + bounding box generation) → Final boxes.**

---

## 3. Available Backbones

We provide 4 backbone options. The heavier ones are more accurate; the lighter ones are faster and smaller — better suited for our deployment targets.

| Backbone | Size | Role |
|---|---|---|
| **VGG-16** | ~138 M params | Heavy baseline, used to validate the pipeline. Not for deployment. |
| **MobileNetV2** | ~3.4 M params | Lightweight. Good accuracy-speed trade-off. |
| **MobileNetV3-Large** | ~5.4 M params | Improved version of V2. Better accuracy for similar cost. |
| **MobileNetV3-Small** | ~2.5 M params | Smallest model. **Best candidate for edge / mobile deployment.** |

All MobileNet backbones are coded from scratch and validated against official PyTorch weights. Pretrained ImageNet weights are loaded automatically — we never train a backbone from zero.

---

## 4. Repository Structure

```
FastCrowdVision/
├── config/                          # YAML configs per backbone variant
│   ├── ssdlite_vgg.yaml
│   ├── ssdlite_mobilenetv2.yaml
│   ├── ssdlite_mobilenetv3large.yaml
│   └── ssdlite_mobilenetv3small.yaml
│
├── ssd.py                  # Main model (SSD & SSDLite classes)
├── mobilenetv2.py          # MobileNetV2 backbone
├── mobilenetv3.py          # MobileNetV3-Large & MobileNetV3-Small backbones
├── l2norm.py               # Normalisation layer (used with VGG only)
│
├── priorbox.py             # Generates candidate boxes across the image
├── detection.py            # Post-processing (filtering + NMS)
├── multiloss.py            # Training loss function
├── utils.py                # Helper functions (IoU, encoding, coordinate conversions)
├── eval.py                 # mAP evaluation
│
├── dataloader.py           # Loads images + YOLO-format labels, splits train/val/test
├── train.py                # Training loop, logging (WandB), checkpoint saving
├── multigpusetup.py        # Multi-GPU support
│
├── SsdTrainingPipelineVOC2007.py   # Main CLI entry point to launch training
├── main.ipynb                      # Notebook for quick experiments
├── voc.sh                          # Downloads PASCAL VOC 2007 dataset
│
├── tests/                  # Unit tests (backbone validation, forward pass, etc.)
├── requirements.txt
└── README.md
```

---

## 5. How to Run Training

### Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# set WANDB_API_KEY, ENTITY, PROJECT in .env
```

### Launch training (not used at inference)
```bash
python SsdTrainingPipelineVOC2007.py \
    <img_dir> \
    <backbone: vgg | mobilenetv2 | mobilenetv3large | mobilenetv3small> \
    <lbl_dir> \
    <nb_classes> \
    <modelname> \
    [--batch_size 20] \
    [--N_epochs 50] \
    [--lr 0.001] \
    [--model_already_trained path/to/checkpoint.pth]
```


Labels must be in **YOLO format** (one `.txt` per image: `class cx cy w h`, normalised). Multi-GPU is handled automatically.

---

## 6. What Is Done ✅

- Full model coding and configuration (all 4 backbone variants)
- Baseline model trained
- Training pipeline with WandB logging, checkpoint saving, multi-GPU support
- Evaluation (mAP metric)
- Dataloader (YOLO-format labels, train/val/test split)
- Unit tests

---

## 7. What Needs to Be Done 🔧

> **⚡ All tasks from 7.1 to 7.4 can be done in parallel — there is no blocking dependency between them.**
> If you are working on the website (7.4) or tracking (7.3), you do **not** need to wait for model strengthening or data augmentation to be finished. Simply pick any pretrained SSDLite model of your choice (e.g. `torchvision.models.detection.ssdlite320_mobilenet_v3_large`) and use its outputs (bounding boxes + scores) with a few lines of code. You can plug in our custom-trained model later when it is ready.
>
> **Only 7.5 (Docker) is blocked** — it should be done once all the other code is finalised.

**Pick your task(s) and write your name below:**

| Task | Owner(s) |
|---|---|
| 7.1 Model Strengthening | Artur |
| 7.2 Data Augmentation & Dataset & S3 | *write your name here* |
| 7.3 Tracking | *write your name here* |
| 7.4 Inference — JS Web App | *write your name here* |
| 7.5 Docker Image | *everyone, after all code is done* |

### 7.1 Model Strengthening 
- Improve model accuracy (hyperparameter tuning, training schedule experiments, architectural tweaks)
- Benchmark backbone variants on the target dataset and pick the best accuracy/speed trade-off

### 7.2 Data Augmentation and Dataset Selection — People on CCTV Cameras and S3 Storage of Final Datasets
- Add training-time image augmentations based on the SSD paper 
- Currently the dataloader only resizes and normalises images; augmentations need to be added
- Find and curate datasets specific to pedestrian / person detection from surveillance cameras (e.g. COCO "person" subset, CrowdHuman, WiderPerson, MOT benchmarks, VisDrone)
- Convert annotations to YOLO format

### 7.3 Tracking
- Add tracking to detections for counting people, the teammate can start with SORT algorithm 
- **No blocker**: use any pretrained SSDLite model to get bounding boxes and build tracking on top of that

### 7.4 Inference — Local Web Application in JavaScript
- Export the trained model to a JS-compatible format (e.g. ONNX → ONNX Runtime Web)
- Build a local web UI that loads the model, accepts video input, and draws bounding boxes on detected people in real time
- This is the **minimum viable deployment target**
- **No blocker**: use any pretrained SSDLite model to prototype the site; swap in our final model later

### 7.5 Docker Image (is done only after we have all code done, can be done by all teammates)
- Package  inference environment ONLY server into a Docker image


---

## 8. Reference Papers

| Topic | Link |
|---|---|
| SSD (model architecture) | [arXiv 1512.02325](https://arxiv.org/abs/1512.02325) |
| MobileNetV2 | [arXiv 1801.04381](https://arxiv.org/abs/1801.04381) |
| MobileNetV3 | [arXiv 1905.02244](https://arxiv.org/abs/1905.02244) |
