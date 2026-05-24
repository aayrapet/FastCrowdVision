import os
import sys

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.mobilenetv3 import MobileNetV3Large
from model.ssd import SSDLite, SSDOnnxWrapper
from training.eval import load_model

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_loaded_model(device: torch.device):
    mn = MobileNetV3Large(0.1, 1000, 1280)
    backbone = mn.features[:-1]

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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    weights_path = hf_hub_download(
        repo_id="aayrapet/SsdFastCrowdVision",
        filename="SSD_FastCrowdVision_v1.pth",
    )
    model, _, _, max_map, _ = load_model(weights_path, device, model, optimizer)
    model.eval()
    return model, max_map


def export(device: torch.device):
    output_path = os.path.join(project_root, "SSD_FastCrowdVision_v1.onnx")
    model, max_map = build_loaded_model(device)
    wrapper = SSDOnnxWrapper(model).eval()

    print(f"Loaded weights — mAP on WiderPeople: {max_map}")

    # random input ensures NMS path is traced during export
    dummy = torch.randn(1, 3, 300, 300, device=device)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["detections"],
        opset_version=12,
        dynamo=False,
    )
    print(f"Exported ONNX model (with NMS) to {output_path}")


if __name__ == "__main__":
    export(torch.device("cpu"))
