import yaml
from ssd import SSD, SSDLite
from l2norm import L2norm
from multigpusetup import ddp_setup
from torch.distributed import init_process_group, destroy_process_group
from train import train
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
import argparse
from train import load_model
from dataloader import DataGeneralLoader,DataSSD300
from dataloader import random_split

parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch"
)

parser.add_argument(
    "img_dir", type=str, help="type folder path  with images in jpef/png format "
)

parser.add_argument(
    "backbone",
    type=str,
    choices=["vgg", "mobilenetv2", "mobilenetv3large", "mobilenetv3small"],
    default="vgg",
    help="backbone to use in SSD, note that all non VGG backbones are only available for SSDLite"
)

parser.add_argument(
    "lbl_dir",
    type=str,
    help="select Yolo-style labels folder path (each file of label per image is txt file )",
)

parser.add_argument(
    "nb_classes", type=int, help="number of classes +1 (background) for your dataset"
)

parser.add_argument(
    "modelname",
    type=str,
    help="unique model name  for file saving (e.g. ssd_coco128V1)",
)


parser.add_argument(
    "--gt_normalised",
    default=True,
    type=bool,
    help="in your dataset are ground truth boxes in labels already normalised between 0 and 1?",
)

parser.add_argument(
    "--batch_size",
    default=20,
    type=int,
    help="select batch size during training, has to be divisible by number of gpus",
)

parser.add_argument(
    "--test_size", default=0.15, type=float, help="prct of dataset used for test set  "
)

parser.add_argument(
    "--gamma", default=0.1, type=float, help="Gamma update for SGD optimizer "
)

parser.add_argument(
    "--val_size",
    default=0.15,
    type=float,
    help="prct of dataset used for validation set  ",
)

parser.add_argument(
    "--lr_schedule_epochs",
    default=[70, 90, 100],
    nargs=3,
    type=int,
    help="at these epochs lr rate will change",
)

parser.add_argument(
    "--alpha",
    default=1.0,
    type=float,
    help="alpha value used for training of ssd, refer to article for more details  ",
)

parser.add_argument(
    "--prob_thr",
    default=0.01,
    type=float,
    help="prob_thr value used for training of ssd, refer to article for more details  ",
)
parser.add_argument(
    "--nms_thr",
    default=0.45,
    type=float,
    help="nms_thr value used for training of ssd, refer to article for more details  ",
)
parser.add_argument(
    "--top_k",
    default=200,
    type=int,
    help="top_k value used for training of ssd, refer to article for more details  ",
)


parser.add_argument(
    "--variances",
    default=[0.1, 0.2],
    nargs=2,
    type=float,
    help="variances for prior box encoding",
)


parser.add_argument(
    "--N_epochs", default=50, type=int, help="number of epochs used for training"
)

parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument(
    "--weight_decay", default=0.0005, type=float, help="weight decay for optimizer"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, help="momentum for SGD optimizer"
)


parser.add_argument(
    "--model_already_trained", default=None, type=str, help="path to model already trained, can be used to continue training"
)

args = parser.parse_args()


# Supports VGG (SSD) and MobileNetV2/V3Large/V3Small (SSDLite) backbones
def pipeline(rank: int, nb_gpus: int, base):

    if nb_gpus == 0:
        device = torch.device("cpu")
    elif nb_gpus == 1:
        device = torch.device("cuda:0")
    elif nb_gpus > 1:
        ddp_setup(rank, nb_gpus)
        device = torch.device(f"cuda:{rank}")
    else:
        raise ValueError("no nb gpus specified")



    images_link = sorted(glob.glob(args.img_dir + "/*.jpg"))
    labels_link = sorted(glob.glob(args.lbl_dir + "/*.txt"))

    train_links,val_links,test_links=random_split(images_link, labels_link, test_size=args.test_size, val_size=args.val_size, seed=None)

    
    trainloader = DataSSD300(
            **train_links, mode="train", gt_normalised=args.gt_normalised
    )
    valloader = DataSSD300(
            **val_links, mode="test", gt_normalised=args.gt_normalised
    )

    testloader = DataSSD300(
            **test_links, mode="test", gt_normalised=args.gt_normalised
    )


    GeneralLoader = DataGeneralLoader(
            batch_size=args.batch_size,
            multigpu=True if nb_gpus > 1 else False,
    )
    train_dataloader, val_dataloader, test_dataloader = GeneralLoader(trainloader,valloader,testloader)

    common_kwargs = dict(
        nb_classes=args.nb_classes,
        phase="train",
        alpha=args.alpha,
        prob_thr=args.prob_thr,
        nms_thr=args.nms_thr,
        top_k=args.top_k,
        variances=args.variances,
        N_epochs=args.N_epochs,
        device=device,
    )

    if args.backbone == "vgg":
        with open("config/ssdlite_vgg.yaml", "r") as f:
            priorbox_config = yaml.safe_load(f)
        model = SSD(
            backbone=base,
            c4_name="22",
            priorbox_config=priorbox_config,
            c4_norm=L2norm(512, 20),
            **common_kwargs,
        ).to(device)
    else:
        config_paths = {
            "mobilenetv2": "config/ssdlite_mobilenetv2.yaml",
            "mobilenetv3large": "config/ssdlite_mobilenetv3large.yaml",
            "mobilenetv3small": "config/ssdlite_mobilenetv3small.yaml",
        }
        c4_names = {
            "mobilenetv2": "8.features.0.features.2",
            "mobilenetv3large": "7.features.1.features.2",
            "mobilenetv3small": "6.features.1.features.2",
        }
        model = SSDLite(
            backbone_config_path=config_paths[args.backbone],
            backbone=base,
            c4_name=c4_names[args.backbone],
            **common_kwargs,
        ).to(device)
    epoch = 0

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    max_map=0
    wandbid=None
    if args.model_already_trained is not None:
        try:
            #attention i suppose models are compatible in all other hyperparameters
            #i will propose a function to check if models are compatible in all other hyperparameters later
            model_loaded, epoch_loaded, optimizer_loaded, max_map_loaded, wandbid_loaded = load_model(args.model_already_trained, device, model, optimizer)
            model = model_loaded
            epoch = epoch_loaded+1
            optimizer = optimizer_loaded
            max_map = max_map_loaded
            wandbid=wandbid_loaded
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting from scratch,sorry")
            
    
    train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        modelname=args.modelname,
        gamma=args.gamma,
        lr_schedule_epochs=args.lr_schedule_epochs,
        start_epoch=epoch,
        max_map=max_map,
        wandbid=wandbid
    )
    if nb_gpus > 1:
        destroy_process_group()


if __name__ == "__main__":
    if args.backbone == "vgg":
        from torchvision.models import VGG16_Weights
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        backbone = vgg[:30]  # until 5_3 layer

    elif args.backbone == "mobilenetv2":
        from torchvision.models import MobileNet_V2_Weights
        from mobilenetv2 import MobileNetV2
        weights = MobileNet_V2_Weights.DEFAULT
        state_dict = weights.get_state_dict()
        
        model = MobileNetV2(0.1, 1000)


        new_state_dict = {}
        for my_key, pretrained_key in zip(model.state_dict().keys(), state_dict.keys()):
            new_state_dict[my_key] = state_dict[pretrained_key]
        model.load_state_dict(new_state_dict)
        backbone = model.features

    elif args.backbone == "mobilenetv3large":
        from torchvision.models import MobileNet_V3_Large_Weights
        from mobilenetv3 import MobileNetV3Large
        weights = MobileNet_V3_Large_Weights.DEFAULT
        state_dict = weights.get_state_dict()
        model = MobileNetV3Large(0.1, 1000, 1280)

        new_state_dict = {}
        for my_key, pretrained_key in zip(model.state_dict().keys(), state_dict.keys()):
            new_state_dict[my_key] = state_dict[pretrained_key]
        model.load_state_dict(new_state_dict)
        backbone=model.features[:-1]
    elif args.backbone == "mobilenetv3small":
        from torchvision.models import MobileNet_V3_Small_Weights
        from mobilenetv3 import MobileNetV3Small
        weights = MobileNet_V3_Small_Weights.DEFAULT
        state_dict = weights.get_state_dict()
        model = MobileNetV3Small(0.1, 1000, 1024)
        new_state_dict = {}
        for my_key, pretrained_key in zip(model.state_dict().keys(), state_dict.keys()):
            new_state_dict[my_key] = state_dict[pretrained_key]
        backbone=model.features[:-1]
    

    nb_gpus = torch.cuda.device_count()
    nb_classes = 21
    if nb_gpus > 1:
        mp.spawn(pipeline, args=(nb_gpus, backbone), nprocs=nb_gpus)
    elif nb_gpus == 1:
        pipeline(None, 1, backbone)
    else:
        pipeline(None, 0, backbone)
