from utils import center_to_corner
from PIL import Image

import torch

import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
from torchvision import tv_tensors

class DataSSD300(torch.utils.data.Dataset):
    """
    Load resized to 300*300 resolution image

    return:
        img_tensor tensor of resized image
        label_list tensor of labels
        gt_box tensor of gt boxes

    rem:
        len(gt_box)=len(label_list)

    """

    def __init__(self, img_dir : list[str], lbl_dir : list[str],mode, gt_normalised: bool = True,trials=10):
        self.images = img_dir
        self.labels = lbl_dir
        if mode =="train":
            #for training set we use data augmentations, but not for test set
            self.transform = v2.Compose(
                [
                    
                    v2.RandomIoUCrop(min_scale = 0.3, max_scale  = 1.0, min_aspect_ratio = 0.5, max_aspect_ratio= 2.0, sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],trials=trials),
                    #https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.SanitizeBoundingBoxes.html
                    v2.ClampBoundingBoxes(),
                    v2.SanitizeBoundingBoxes(),
                    v2.RandomHorizontalFlip(p=0.5),
                    #https://arxiv.org/pdf/1312.5402
                    v2.RandomPhotometricDistort(
                        brightness=(0.5, 1.5),
                        contrast=(0.5, 1.5),
                        saturation=(0.5, 1.5),
                    ),

                    v2.Resize((300, 300)),
                    v2.ToImage(),  
                    v2.ToDtype(torch.float32, scale=True),
                    #we suppose backbones were pretrained on image net 
                    v2.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Resize((300, 300)),
                    v2.ToImage(),  
                    v2.ToDtype(torch.float32, scale=True),
                    #we suppose backbones were pretrained on image net 
                    #we take their stat
                    v2.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        #https://docs.pytorch.org/vision/main/auto_examples/transforms/plot_transforms_getting_started.html
        self.gt_normalised = gt_normalised

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")#RGBA->RGB force 

        W_mult, H_mult = img.size if self.gt_normalised else (1,1)
        W,H=img.size 
     
        # labels and gt boxes easy extraction
        with open(self.labels[idx]) as f:
            gt_box = []
            label_list = []
            for line in f:
                label, cx, cy, w, h = map(float, line.split())
                #when normalised gt boxes coords ( in (0;1)) need to get actual coords FOR FURTHER TRANSFORMATION
                gt_box.append((cx*W_mult, cy*H_mult , w*W_mult , h*H_mult ))

                label_list.append(
                    label + 1
                )  # yolo labels start at 0, my ssd start at 1, O is BG
            gt_box = center_to_corner(torch.tensor(gt_box, dtype=torch.float32))
         
            label_list = torch.tensor(label_list, dtype=torch.int64)

        #https://docs.pytorch.org/vision/stable/transforms.html
        boxes = tv_tensors.BoundingBoxes(gt_box, format="XYXY", canvas_size=(H,W))

        img, target = self.transform(img, {"boxes": boxes, "labels": label_list})
        boxes = target["boxes"]
        labels = target["labels"]
        

        return img, labels, boxes/300


class DataGeneralLoader(nn.Module):
    """
    Split into training, validation, testing dataloaders, specifying general parameters 
    """

    def __init__(
        self, batch_size: int, multigpu=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.multigpu = multigpu

    @staticmethod
    def collate_ssd(batch):
        images, labels, boxes = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, list(labels), list(boxes)

    def forward(self, dataset_train: DataSSD300, dataset_eval: DataSSD300, dataset_test: DataSSD300):
        # https://stackoverflow.com/questions/65138643/examples-or-explanations-of-pytorch-dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=False if self.multigpu else True,
            collate_fn=self.collate_ssd,
            #https://docs.pytorch.org/tutorials/beginner/ddp_series_multigpu.html
            sampler=DistributedSampler(dataset_train) if self.multigpu else None,
            # persistent_workers=True,
            # # num_workers=4,
            # pin_memory=True

        )
        val_dataloader = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=self.batch_size,
            
            collate_fn=self.collate_ssd,
            sampler=DistributedSampler(dataset_eval) if self.multigpu else None,
           
        )
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self.batch_size,
           
            collate_fn=self.collate_ssd,
            sampler=DistributedSampler(dataset_test) if self.multigpu else None,
        
        )

        return train_dataloader, val_dataloader, test_dataloader

import random

def random_split(images_link, labels_link, test_size=0.15, val_size=0.15, seed=None):

    if len(images_link) != len(labels_link):
        raise ValueError("Images and labels must have the same length")

    if seed is not None:
        random.seed(seed)
    data = list(zip(images_link, labels_link))
    random.shuffle(data)

    n = len(data)
    test_amount = int(n * test_size)
    val_amount = int(n * val_size)

    test_links = data[:test_amount]
    val_links = data[test_amount:test_amount + val_amount]
    train_links = data[test_amount + val_amount:]

    def unzip(dataset):
        x, y = zip(*dataset)
        return {"img_dir" : list(x), "lbl_dir" : list(y) }

    train_links = unzip(train_links)
    val_links = unzip(val_links)
    test_links = unzip(test_links)

    return train_links,val_links,test_links