# inference.py — Import-safe model loading and per-frame detection for the server.
# Separates model setup from CLI/argparse concerns in SsdFastCrowdVision.py.

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from mobilenetv3 import MobileNetV3Large
from ssd import SSDLite
from transforms import test_val_transform
from eval import load_model
from torchvision.transforms import v2
import os
import yaml
import time

# project root = folder containing this file
project_root = os.path.dirname(os.path.abspath(__file__))


def load_ssd_model(device):
    """Build SSDLite + MobileNetV3 backbone, download weights from HuggingFace,
    return (model, config_dict, transform) ready for inference."""

    # build MobileNetV3Large backbone without the final avg-pool + classifier
    mn = MobileNetV3Large(0.1, 1000, 1280)
    backbone = mn.features[:-1]

    # build SSDLite on top — use absolute path for the YAML config
    # so the server works regardless of working directory
    model = SSDLite(
        backbone_config_path=os.path.join(project_root, "config", "ssdlite_mobilenetv3large.yaml"),
        backbone=backbone,
        c4_name="7.features.1.features.2",
        nb_classes=4,
        phase="test",
        alpha=1.0,
        prob_thr=0.01,
        nms_thr=0.45,
        top_k=200,
        variances=[0.1, 0.2],
        N_epochs=50,
        device=device,
    ).to(device)

    # dummy optimizer — load_model() requires one to restore checkpoint state
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    # download weights from HuggingFace (cached locally after first download)
    weights_path = hf_hub_download(
        repo_id="aayrapet/SsdFastCrowdVision",
        filename="SSD_FastCrowdVision_v1.pth",
    )

    # load trained weights into the model
    model, _, _, max_map, _ = load_model(weights_path, device, model, optimizer)
    model.eval()
    model.phase = "test"
    print(f"Model loaded — mAP on WiderPeople: {max_map}")

    # load class names (1: pedestrians, 2: riders, 3: partially-visible persons)
    wider_yaml = os.path.join(project_root, "datasets", "WiderPeople", "widerpeople.yaml")
    with open(wider_yaml) as f:
        config = yaml.safe_load(f)["names"]

    # test transform: resize to 300x300, convert to float, normalize with ImageNet stats
    transform = v2.Compose(test_val_transform)

    return model, config, transform


def detect_frame(model, pil_image, transform, device, score_thr=0.25):
    """Run SSD on a single PIL image.

    Returns np.ndarray of shape (N, 6) — columns: [x1, y1, x2, y2, score, class].
    Coordinates are in pixels (original image size).
    Returns empty (0, 6) array if no detections pass the threshold.
    """
    t0 = time.perf_counter()
    W, H = pil_image.size

    # transform PIL image to 300x300 normalized tensor, add batch dim
    x = transform(pil_image).unsqueeze(0).to(device)

    # SSD forward pass + NMS (no gradients needed at inference)
    with torch.no_grad():
        _, _, topk = model(x)

    # remove batch dimension: (1, top_k, 6) → (top_k, 6)
    topk = topk.squeeze(0)

    # keep only detections above the score threshold
    mask = topk[:, 4] > score_thr
    if not mask.any():
        return np.empty((0, 6))

    topk = topk[mask]

    # scale normalized [0,1] box coordinates to pixel coordinates
    topk[:, [0, 2]] *= W
    topk[:, [1, 3]] *= H

    elapsed = time.perf_counter() - t0
    print(f"[inference] {elapsed*1000:.1f} ms")

    return topk.cpu().numpy()
