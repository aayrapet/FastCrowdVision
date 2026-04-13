import torch
import time 
from huggingface_hub import hf_hub_download
from mobilenetv3 import MobileNetV3Large
from ssd import SSDLite
from transforms import test_val_transform
from eval import load_model,predict
import argparse
from PIL import Image
from torchvision.transforms import v2
import os
from draw_inference import draw_boxes_with_labels
cwd=os.getcwd()
from IPython.display import display

import yaml

transform = v2.Compose(
                test_val_transform               
)
#independently from where you are, project root is this
project_root = cwd.split("FastCrowdVision")[0] + "FastCrowdVision"

#get names of labels
wider_dir=os.path.join(project_root, "datasets", "WiderPeople","widerpeople.yaml")

with open(wider_dir) as f:
    cfg = yaml.safe_load(f)
config = cfg["names"] 


parser = argparse.ArgumentParser(
    description="SSD model at inference time"
)

parser.add_argument(
    "image_path", type=str, help="image path file (jpeg/png)"
)

parser.add_argument(
    #https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    "--show_labels", default=True, action=argparse.BooleanOptionalAction,
    help="On image do you want to show labels and probability"
)

parser.add_argument(
    "--score_thr", default=0.25,
    type=float, help="Probability threshold for filtering bboxes on images"
)

args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#this module by default loads model stored on Hugging Face


def predict_image_path(model, img,transform):
    """Returns (locs, confs, detections, x). x is (1,3,300,300) normalized batch for drawing."""
   
    #image is transformed to N,C,H,W 4D is supposed by ssd class
    x = transform(img).unsqueeze(0).to(device)
    _, _, topk = predict(model, x)
    return topk, x

def make_backbone():
    mn = MobileNetV3Large(0.1, 1000, 1280)
    return mn.features[:-1]

def measure_fps_cpu(model, device=torch.device("cpu"), input_size=300, n_warmup=10, n_iters=100):
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
    print(f"~{fps:.2f} FPS ({n_iters} iters, {input_size}x{input_size}, {device})")
    return fps


if __name__=="__main__":
    backbone_w = make_backbone()   
    modelW = SSDLite(
        backbone_config_path="config/ssdlite_mobilenetv3large.yaml",
        backbone=backbone_w,
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
    optimizerW = torch.optim.Adam(
        modelW.parameters(),
        lr=0.001,
        weight_decay=0.0005,
    )
    n_params = sum(p.numel() for p in modelW.parameters())

    print(f" Model number of params: {n_params:,} ")


    path = hf_hub_download(
        repo_id="aayrapet/SsdFastCrowdVision",
        filename="SSD_FastCrowdVision_v1.pth"
    )

    modelW , _, _,maxmapW,_=load_model(path, device, modelW, optimizerW)
    print("map on WiderPeople dataset:",maxmapW)
    
    # import time
# import torch



# usage after model is loaded
    measure_fps_cpu(modelW)

    img = Image.open(args.image_path).convert("RGB")#RGBA->RGB force 
    W,H=img.size 
  
    #get model detections : ssd forward pass + NMS
    topk,x=predict_image_path(modelW,  img,transform)
    out=draw_boxes_with_labels(
    img,
    config,
    topk,
    H,W,
    score_thr= args.score_thr,
    show_scores= args.show_labels)
    out.save("result.jpg")

    
    


