import torch

from ssd import SSDLite
import time 
from mobilenetv3 import MobileNetV3Small,MobileNetV3Large

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nb_classes = 21  

def measure_fps_cpu(model, model_name,device=torch.device("cpu"), input_size=300, n_warmup=10, n_iters=100):
    model.eval()
    model.phase = "test"
    model.to(device)

    x = torch.randn(1, 3, input_size, input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(x)
        elapsed = time.perf_counter() - t0

    fps = n_iters / elapsed
    print(f"~{fps:.2f} FPS ({n_iters} iters, {input_size}x{input_size}, {device})", model_name)
    return fps


#------------Small--------

model = MobileNetV3Small(0.1, 1000, 1024)
backbone=model.features[:-1]


model = SSDLite(
    backbone_config_path="config/ssdlite_mobilenetv3small.yaml",
    backbone=backbone,
    c4_name="6.features.1.features.2",
    nb_classes=nb_classes,
    phase="test",
    alpha=1.0,
    prob_thr=0.01,
    nms_thr=0.45,
    top_k=200,
    variances=[0.1, 0.2],
    N_epochs=50,
    device=device,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0005,
)

nb_parameters=0
for param in model.parameters():
    nb_parameters=nb_parameters+param.numel()
print("Number parameters of MobileNetV3Small:",  nb_parameters)

measure_fps_cpu(model,"MobileNetV3Small")


#------------Large--------


model = MobileNetV3Large(0.1, 1000, 1280)
backbone=model.features[:-1]

model = SSDLite(
    backbone_config_path="config/ssdlite_mobilenetv3large.yaml",
    backbone=backbone,
    c4_name="7.features.1.features.2",
    nb_classes=nb_classes,
    phase="test",
    alpha=1.0,
    prob_thr=0.01,
    nms_thr=0.45,
    top_k=200,
    variances=[0.1, 0.2],
    N_epochs=50,
    device=device,
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0005,
)

nb_parameters=0
for param in model.parameters():
    nb_parameters=nb_parameters+param.numel()
print("Number parameters of MobileNetV3Large:",  nb_parameters)

measure_fps_cpu(model,"MobileNetV3Large")

