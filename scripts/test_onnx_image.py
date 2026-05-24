import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.draw_inference import draw_boxes_with_labels
from serving.inference import load_ssd_model_onnx

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Test ONNX SSD on one image")
    parser.add_argument("image_path", type=str, help="Path to input image (jpeg/png)")
    parser.add_argument(
        "--output",
        default=os.path.join(project_root, "result_onnx.jpg"),
        help="Path for annotated output image",
    )
    parser.add_argument(
        "--score_thr",
        default=0.25,
        type=float,
        help="Probability threshold for filtering boxes",
    )
    parser.add_argument(
        "--show_labels",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Draw class labels and scores on boxes",
    )
    args = parser.parse_args()

    session, config, transform = load_ssd_model_onnx()

    img = Image.open(args.image_path).convert("RGB")
    W, H = img.size

    x = transform(img).unsqueeze(0).numpy()
    topk = torch.from_numpy(session.run(["detections"], {"image": x})[0])
    print(f"Detections above threshold: {(topk[0, :, 4] > args.score_thr).sum().item()}")

    out = draw_boxes_with_labels(
        img,
        config,
        topk,
        H,
        W,
        score_thr=args.score_thr,
        show_scores=args.show_labels,
    )
    out.save(args.output)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
