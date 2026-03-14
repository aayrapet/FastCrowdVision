#build SSD per backbone and run one forward pass
import os
import sys
import yaml
import torch
import torch.nn as nn
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ssd import SSD, SSDLite
from l2norm import L2norm
from torchvision import models
from torchvision.models import VGG16_Weights, MobileNet_V2_Weights, MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights
from mobilenetv2 import MobileNetV2
from mobilenetv3 import MobileNetV3Large, MobileNetV3Small

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
BATCH = 2
NB_CLASSES = 21
DEVICE = torch.device("cpu")


def _load_priorbox(name):
    path = os.path.join(CONFIG_DIR, f"ssdlite_{name}.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_backbone_vgg():
    #i dont code vgg myself, i just load pretrained weigts
    #reason : vgg architecture is too easy to code and i dont want to waste time on it
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    vgg[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    return vgg[:30]


def _get_backbone_mobilenetv2():
    #note how i do : i first code class by hand and then load pretrained weights on image net from pytorch 
    #i code them myself for learning purposes
    model = MobileNetV2(0.1, 1000)
    state_dict = MobileNet_V2_Weights.DEFAULT.get_state_dict()
    new_state_dict = {mk: state_dict[pk] for mk, pk in zip(model.state_dict().keys(), state_dict.keys())}
    model.load_state_dict(new_state_dict)
    return model.features


def _get_backbone_mobilenetv3large():
    model = MobileNetV3Large(0.1, 1000, 1280)
    state_dict = MobileNet_V3_Large_Weights.DEFAULT.get_state_dict()
    new_state_dict = {mk: state_dict[pk] for mk, pk in zip(model.state_dict().keys(), state_dict.keys())}
    model.load_state_dict(new_state_dict)
    return model.features[:-1]


def _get_backbone_mobilenetv3small():
    model = MobileNetV3Small(0.1, 1000, 1024)
    state_dict = MobileNet_V3_Small_Weights.DEFAULT.get_state_dict()
    new_state_dict = {mk: state_dict[pk] for mk, pk in zip(model.state_dict().keys(), state_dict.keys())}
    model.load_state_dict(new_state_dict)
    return model.features[:-1]


COMMON_KWARGS = dict(
    nb_classes=NB_CLASSES,
    phase="train",
    alpha=1.0,
    prob_thr=0.01,
    nms_thr=0.45,
    top_k=200,
    variances=[0.1, 0.2],
    N_epochs=50,
    device=DEVICE,
)


def test_ssd_vgg_forward():
    backbone = _get_backbone_vgg()
    priorbox_config = _load_priorbox("vgg")
    model = SSD(
        backbone=backbone,
        c4_name="22",
        priorbox_config=priorbox_config,
        c4_norm=L2norm(512, 20),
        **COMMON_KWARGS,
    )
    x = torch.randn(BATCH, 3, 300, 300)
    locs, confs = model(x)
    assert locs.shape == (BATCH, model.anchors.shape[0], 4)
    assert confs.shape == (BATCH, model.anchors.shape[0], NB_CLASSES)


def test_ssdlite_mobilenetv2_forward():
    backbone = _get_backbone_mobilenetv2()
    config_path = os.path.join(CONFIG_DIR, "ssdlite_mobilenetv2.yaml")
    model = SSDLite(
        backbone_config_path=config_path,
        backbone=backbone,
        c4_name="8.features.0.features.2",
        **COMMON_KWARGS,
    )
    x = torch.randn(BATCH, 3, 300, 300)
    locs, confs = model(x)
    assert locs.shape == (BATCH, model.anchors.shape[0], 4)
    assert confs.shape == (BATCH, model.anchors.shape[0], NB_CLASSES)


def test_ssdlite_mobilenetv3large_forward():
    backbone = _get_backbone_mobilenetv3large()
    config_path = os.path.join(CONFIG_DIR, "ssdlite_mobilenetv3large.yaml")
    model = SSDLite(
        backbone_config_path=config_path,
        backbone=backbone,
        c4_name="7.features.1.features.2",
        **COMMON_KWARGS,
    )
    x = torch.randn(BATCH, 3, 300, 300)
    locs, confs = model(x)
    assert locs.shape == (BATCH, model.anchors.shape[0], 4)
    assert confs.shape == (BATCH, model.anchors.shape[0], NB_CLASSES)


def test_ssdlite_mobilenetv3small_forward():
    backbone = _get_backbone_mobilenetv3small()
    config_path = os.path.join(CONFIG_DIR, "ssdlite_mobilenetv3small.yaml")
    model = SSDLite(
        backbone_config_path=config_path,
        backbone=backbone,
        c4_name="6.features.1.features.2",
        **COMMON_KWARGS,
    )
    x = torch.randn(BATCH, 3, 300, 300)
    locs, confs = model(x)
    assert locs.shape == (BATCH, model.anchors.shape[0], 4)
    assert confs.shape == (BATCH, model.anchors.shape[0], NB_CLASSES)
