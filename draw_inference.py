import torch
from transforms import means, stds
from PIL import Image
from torchvision.transforms import v2
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
#normalisation coefficients for each channel
IMAGENET_MEAN = torch.tensor(means).view(3, 1, 1)
IMAGENET_STD = torch.tensor(stds).view(3, 1, 1)


def draw_boxes_with_labels(
    config,
    topk_bboxes,
    transformed_img_tensor,
    H,W,
    score_thr: float = 0.25,
    show_scores: bool = False,
    

):
    """Filter detections by score_thr, draw bboxes detected , returns image"""

    transform_original=v2.Resize((H,W))
    #get model detections : ssd forward pass + NMS
    topk,x=topk_bboxes,transformed_img_tensor
  
    #only one image at a time
    topk=topk.squeeze(0)
    x=x.squeeze(0)
    #filter 
    indexes=topk[:,4]>score_thr
    #restore pixels
    img_vis=x*IMAGENET_STD+ IMAGENET_MEAN
    #if no bboxes then just plot image 
    if torch.all(indexes==False).item():
        return to_pil_image(img_vis)
    
    #filter >thr boxes
    topk=topk[indexes,:]
    scores = topk[:, 4]
    cls = topk[:, 5].long()
    topk=topk[:,:4]
    #resize to original image from 300*300 SSD output 
    img_vis = transform_original(img_vis)
    topk[:, [0, 2]]= topk[:, [0, 2]]*W
    topk[:, [1, 3]] = topk[:, [1, 3]]*H

    if show_scores:
        labels_str = [
            f"{config[int(c.item())]} {scores[i].item():.2f}"
            for i, c in enumerate(cls)
        ]
        #https://docs.pytorch.org/vision/stable/generated/torchvision.utils.draw_bounding_boxes.html
        drawn = draw_bounding_boxes(img_vis, topk, labels=labels_str, width=2)
    else:
        drawn = draw_bounding_boxes(img_vis, topk,  width=2)

    return to_pil_image(drawn)