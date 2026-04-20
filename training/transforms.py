
import torch 
from torchvision.transforms import v2

means=[0.485, 0.456, 0.406]
stds=[0.229, 0.224, 0.225]

train_transform=[
                    
                    v2.RandomIoUCrop(min_scale = 0.3, max_scale  = 1.0, min_aspect_ratio = 0.5, max_aspect_ratio= 2.0, sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],trials=10),
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
                            mean=means,
                            std=stds
                    )
                    
]


test_val_transform=[
                    v2.Resize((300, 300)),
                    v2.ToImage(),  
                    v2.ToDtype(torch.float32, scale=True),
                    #we suppose backbones were pretrained on image net 
                    #we take their stat
                    v2.Normalize(
                        mean=means,
                        std=stds
                    ),
]