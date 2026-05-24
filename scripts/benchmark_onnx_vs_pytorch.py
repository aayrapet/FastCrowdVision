import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving.inference import load_ssd_model, load_ssd_model_onnx

N_WARMUP = 10
N_ITERS = 100
DEVICE = torch.device("cpu")


def _benchmark_pytorch(model, x, n_warmup, n_iters):
    model.eval()
    model.phase = "test"

    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model(x)
        elapsed = time.perf_counter() - t0

    return elapsed / n_iters


def _benchmark_onnx(session, x_np, n_warmup, n_iters):
    for _ in range(n_warmup):
        session.run(["detections"], {"image": x_np})

    t0 = time.perf_counter()
    for _ in range(n_iters):
        session.run(["detections"], {"image": x_np})
    elapsed = time.perf_counter() - t0

    return elapsed / n_iters


def _print_result(name, mean_s, baseline_s=None):
    fps = 1.0 / mean_s
    ms = mean_s * 1000
    line = f"{name:8s}  {ms:7.2f} ms/frame  ({fps:6.1f} FPS)"
    if baseline_s is not None and mean_s > 0:
        speedup = baseline_s / mean_s
        line += f"  — {speedup:.2f}x vs PyTorch"
    print(line)


def main():
    #test on random image 
    print("Input: random tensor (1, 3, 300, 300)")
    #load both models from hf, compare .pth model vs onnx model
    model, _, _ = load_ssd_model(DEVICE)
    session, _, _ = load_ssd_model_onnx()

    x = torch.randn(1, 3, 300, 300)
    x_onnx = x.numpy()

    pytorch_ms = _benchmark_pytorch(model, x, N_WARMUP, N_ITERS)
    onnx_ms = _benchmark_onnx(session, x_onnx, N_WARMUP, N_ITERS)

    _print_result("PyTorch", pytorch_ms)
    _print_result("ONNX", onnx_ms, baseline_s=pytorch_ms)


if __name__ == "__main__":
    main()
