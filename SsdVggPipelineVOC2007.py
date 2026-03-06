from ssd import SSD
from multigpusetup import ddp_setup
from torch.distributed import init_process_group, destroy_process_group
from train import train
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from dataloader import DataSSD300, DataSplitter
import argparse
from train import load_model

parser = argparse.ArgumentParser(
    description="Single Shot MultiBox Detector Training With Pytorch"
)


parser.add_argument(
    "img_dir", type=str, help="type folder path  with images in jpef/png format "
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


# fo the moment we have only SSDVGG architecture, we will be able to select in argparse other archi soon
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

    someloader = DataSSD300(
        args.img_dir, args.lbl_dir, gt_normalised=args.gt_normalised
    )
    splitter = DataSplitter(
        batch_size=args.batch_size,
        test_size=args.test_size,
        val_size=args.val_size,
        multigpu=True if nb_gpus > 1 else False,
    )
    train_dataloader, val_dataloader, test_dataloader = splitter(someloader)

   
    model = SSD(
        base,
        nb_classes=args.nb_classes,
        phase="train",
        alpha=args.alpha,
        prob_thr=args.prob_thr,
        nms_thr=args.nms_thr,
        top_k=args.top_k,
        variances=args.variances,
        N_epochs=args.N_epochs,
        device=device,
    ).to(device)
    epoch = 0

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    best_val_loss = float("inf")
    wandbid=None
    if args.model_already_trained is not None:
        try:
            #attention i suppose models are compatible in all other hyperparameters
            #i will propose a function to check if models are compatible in all other hyperparameters later
            model_loaded, epoch_loaded, optimizer_loaded, best_val_loss_loaded, wandbid_loaded = load_model(args.model_already_trained, device, model, optimizer)
            model = model_loaded
            epoch = epoch_loaded
            optimizer = optimizer_loaded
            best_val_loss = best_val_loss_loaded
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
        best_val_loss=best_val_loss,
        wandbid=wandbid
    )
    if nb_gpus > 1:
        destroy_process_group()


if __name__ == "__main__":

    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    vgg[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    base = nn.ModuleList(vgg[:30])  # until 5_3 layer

    nb_gpus = torch.cuda.device_count()
    nb_classes = 21
    if nb_gpus > 1:
        mp.spawn(pipeline, args=(nb_gpus, base), nprocs=nb_gpus)
    elif nb_gpus == 1:
        pipeline(None, 1, base)
    else:
        pipeline(None, 0, base)
