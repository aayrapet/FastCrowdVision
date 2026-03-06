from ssd import SSD
from multigpusetup import ddp_setup
from torch.distributed import init_process_group, destroy_process_group
from train import train 
import  torch.multiprocessing as mp 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from dataloader import DataSSD300, DataSplitter

def pipeline(rank : int  ,nb_classes : int ,nb_gpus : int,img_dir,lbl_dir ):

    
    if nb_gpus==0:
         device = torch.device("cpu")
    elif nb_gpus==1:
        device = torch.device("cuda:0")
    elif nb_gpus>1:
        ddp_setup(rank, nb_gpus)
        device = torch.device(f"cuda:{rank}")
    else:
        raise ValueError("no nb gpus specified")

    coco128loader=DataSSD300(img_dir,lbl_dir,gt_normalised=True)
    splitter=DataSplitter(batch_size=20,test_size=0.15,val_size=0.15,multigpu = True if nb_gpus>1 else False)
    train_dataloader, val_dataloader, test_dataloader=splitter(coco128loader)


    model=SSD(base,nb_classes=nb_classes,phase="train",alpha=1,device=device,prob_thr=0.01,nms_thr=0.45,top_k=200,variances=(0.1,0.2)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay = 0.0005, momentum = 0.9)
    train(model,optimizer,train_dataloader,val_dataloader,modelname="ssd_coco128V1")
    if nb_gpus>1:
        destroy_process_group()

vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
vgg[16] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
base=nn.ModuleList(vgg[:30])#until 5_3 layer
base


nb_gpus=torch.cuda.device_count()
nb_classes=21
if nb_gpus>1:
    mp.spawn(pipeline,args=(nb_classes,nb_gpus,img_dir,lbl_dir),nprocs=nb_gpus)
elif nb_gpus==1:
    pipeline(None,nb_classes,1,img_dir,lbl_dir)
else:
    pipeline(None,nb_classes,0,img_dir,lbl_dir  )
