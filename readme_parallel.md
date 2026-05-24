# ONNX export 

This project runs object detection algorithm Single Shot Detection using Lightweight Neural Networks such as MobilenetV3 for backbone (2019), the objective is to efficiently track people in dense city areas on videos on CPUs, for this I use detection over frames and simple SORT tracking algorithm using Kalman Filter. For model references, deployment and overall vision, please refer to `README.md`. For now, i use only web for video processing, but in future the objective is to run these algorithms on edge devices, such as mobile phones or cheap Raspberry chips. 

For training, the Wider People Dataset was used for training, next the trained model was uploaded to Hugging Face (https://huggingface.co/aayrapet).

For inference the original project ran  entirely in PyTorch: the server built an SSDLite model, downloaded `SSD_FastCrowdVision_v1.pth` from HuggingFace, and applied post-processing (NMS) through the Python `Detection` module in `model/detection.py`. To speed up I export the model to ONNX. The export script loads the same trained weights from HuggingFace, wraps the network in `SSDOnnxWrapper` (defined in `model/ssd.py`) that is then converted to onnx format. During export, NMS from `model/detection.py` become ONNX's built-in `NonMaxSuppression` operator (https://onnx.ai/onnx/operators/onnx__NonMaxSuppression.html).


On the inference side, `serving/inference.py` now exposes two loaders. `load_ssd_model(device)` is the original PyTorch path, which was used before to do forward pass for image. I added `load_ssd_model_onnx()` that downloads `SSD_FastCrowdVision_v1.onnx` from the same HuggingFace repo (`aayrapet/SsdFastCrowdVision`) and creates an ONNX Runtime session; NMS is already inside the graph. `detect_frame_onnx()` mirrors `detect_frame()` so the rest of the pipeline (tracking, drawing) stays the same. The FastAPI server in `serving/server.py` was switched to the ONNX loader and `detect_frame_onnx`, so Docker and local uvicorn both use the HF ONNX model at startup without needing a local `.pth` file.

New scripts under `scripts/` support the workflow: `export_onnx.py` produces the ONNX model, `test_onnx_image.py` runs one image and saves an annotated result, and `benchmark_onnx_vs_pytorch.py` compares latency between the HF PyTorch and HF ONNX models on CPU with a random tensor. 

## Download a test image to the project root

A common WiderPeople test photo is hosted on SSP Cloud MinIO. To copy it into the repository root as `image000047.jpg`, run from the project folder.

On Windows PowerShell:

```powershell
Invoke-WebRequest -Uri "https://minio.lab.sspcloud.fr/aayrapetyan/FastCrowdVision/datasets/WiderPeople/test/images/image000047.jpg" -OutFile "image000047.jpg"
```

On Linux or macOS:

```bash
curl -L -o image000047.jpg "https://minio.lab.sspcloud.fr/aayrapetyan/FastCrowdVision/datasets/WiderPeople/test/images/image000047.jpg"
```

You can also download the file in a browser and save it manually as `image000047.jpg` in the FastCrowdVision root.

## Export the ONNX model

Export requires  `requirements-api.txt`. Activate your venv.

Then export model to onnx 

```powershell
python scripts/export_onnx.py
```

The script downloads `SSD_FastCrowdVision_v1.pth` from HuggingFace, loads weights into SSDLite, wraps the model with `SSDOnnxWrapper`, and writes `SSD_FastCrowdVision_v1.onnx` in the project root. Export uses the legacy ONNX tracer (`dynamo=False`) because the detection post-processing contains Python control flow in detection that the newer exporter does not handle yet. After export, I uploaded `SSD_FastCrowdVision_v1.onnx` to the HuggingFace repo.

## Test ONNX inference on one image


```powershell
python scripts/test_onnx_image.py image000047.jpg
```

This calls `load_ssd_model_onnx()`, which downloads the ONNX model from HuggingFace, runs inference, and saves `result_onnx.jpg` in the project root with bounding boxes drawn. 

## Benchmark PyTorch vs ONNX

To compare inference speed on CPU with the same random input:

```powershell
python scripts/benchmark_onnx_vs_pytorch.py
```

Both backends load from HuggingFace: `.pth` for PyTorch and `.onnx` for ONNX Runtime. The script prints milliseconds per frame and FPS for each, plus a speedup factor relative to PyTorch. Timing excludes the image transform so the comparison focuses on forward pass and NMS.


## Docker

same as before (refer more to REAMDE.md for deployment)

```powershell
docker build -t fastcrowdvision .
docker run -p 8000:8000 fastcrowdvision
```

## Conclusion

This exercise allowed me to go further after model development and basic deployment and export the model to ONNX, this is interesting because first ONNX does not require pytorch at all to run in applications and it is faster (confirmed with benchmarks, 2.5 faster). However, for non standard neural networks, as in image detection we use here (when there is data post processing after forward pass as NMS), it is better to use legacy ONNC export.

