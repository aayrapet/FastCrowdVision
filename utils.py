import os
import torch
import random
import numpy as np
import math as mt

def setup_device_and_seed(seed=42):
   

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

def calculate_sk(k, m):
    """
    k is index of ft map , m is total nb of ft map
    (ft map are map participating in prediction)
    k=1....M
    """
    smin, smax = 0.2, 0.9
    if k > m:
        return 1
    return smin + (smax - smin) * (k - 1) / (m - 1)


def calculate_anchor_w_h1(sk, a):
    return sk * mt.sqrt(a), sk / mt.sqrt(a)


def calculate_anchor_w_h2(sk, a):
    return sk / mt.sqrt(a), sk * mt.sqrt(a)


def normalised_anchor_coords(i, j, f, w, h):
    """
    i , j in len(feature map)=0...f-1
    w=w_k_a
    h=h_k_a
    """
    centerx = (j + 0.5) / f
    centery = (i + 0.5) / f

    x2 = min(centerx + w / 2, 1)
    y2 = min(centery + h / 2, 1)

    x1 = max(centerx - w / 2, 0)
    y1 = max(centery - h / 2, 0)
    return x1, y1, x2, y2


def normalised_gt_coords(box, H, W):
    """
    normalise gt coords so that they are in [0,1]

    we do so for iou with anchors that by default are in [0,1]
    """

    return torch.cat(
        (
            (box[:, 0] / W).unsqueeze(1),
            (box[:, 1] / H).unsqueeze(1),
            (box[:, 2] / W).unsqueeze(1),
            (box[:, 3] / H).unsqueeze(1),
        ),
        -1,
    )

def _make_divisible(v, divisor, min_value=None):

  """
   this code is taken from the original code of mobilenet pytorch library
  """
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return int(new_v)




def center_to_corner(box):
    """from cx,cy,w,h to x1,y1,x2,y2"""

    return torch.cat(
        (
            box[:, :2] - box[:, 2:4] / 2,
            box[:, :2] + box[:, 2:4] / 2,
        ),
        1,
    )


def corner_to_center_scalar(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def corner_to_center(box):
    """from x1,y1,x2,y2 to cx,cy,w,h"""
    return torch.cat(
        (((box[:, 0:2] + box[:, 2:4]) / 2), (box[:, 2:4] - box[:, 0:2])), dim=1
    )


def iou(anchors, gtboxes):
    """
    gtboxes are in center to corner version i assume for calculation of iou
    anchors are in corner to center version 
    iou standard formula for jaccard overlap

    output: 
        [nb_anchors,nb_gt_boxes] matrix for pairwise iou 

    """

    anchors = center_to_corner(anchors)
    nb_anchors = anchors.shape[0]
    n_gt = gtboxes.shape[0]

    a = torch.min(anchors[:, 2].unsqueeze(1), gtboxes[:, 2].unsqueeze(0))
    b = torch.max(anchors[:, 0].unsqueeze(1), gtboxes[:, 0].unsqueeze(0))
    d = torch.max(anchors[:, 1].unsqueeze(1), gtboxes[:, 1].unsqueeze(0))
    c = torch.min(anchors[:, 3].unsqueeze(1), gtboxes[:, 3].unsqueeze(0))

    h = torch.clamp((a - b), min=0)
    w = torch.clamp((c - d), min=0)
    intersection = h * w

    A_area = (anchors[:, 2].unsqueeze(1) - anchors[:, 0].unsqueeze(1)) * (
        anchors[:, 3].unsqueeze(1) - anchors[:, 1].unsqueeze(1)
    )
    B_area = (gtboxes[:, 2].unsqueeze(1) - gtboxes[:, 0].unsqueeze(1)) * (
        gtboxes[:, 3].unsqueeze(1) - gtboxes[:, 1].unsqueeze(1)
    )
    union = (
        A_area + B_area.unsqueeze(0)[:, :, 0].expand(nb_anchors, n_gt) - intersection
    )

    return intersection / union


def decode(encoded, anchors, variances):
    """
    decode based on article's formulas
    """

    return torch.cat(
        (
            (encoded[:, 0] * anchors[:, 2] * variances[0] + anchors[:, 0]).unsqueeze(1),
            (encoded[:, 1] * anchors[:, 3] * variances[0] + anchors[:, 1]).unsqueeze(1),
            (torch.exp(encoded[:, 2] * variances[1]) * anchors[:, 2]).unsqueeze(1),
            (torch.exp(encoded[:, 3] * variances[1]) * anchors[:, 3]).unsqueeze(1),
        ),
        -1,
    )


def encode(matches_anchors, anchors, variances):
    """
    encode matched_anchors  based on article's formulas.
    This is done for training
    works on center to corner version
    cx cy w h  
    """

    return torch.cat(
        (
            (
                (matches_anchors[:, 0] - anchors[:, 0]) / (anchors[:, 2] * variances[0])
            ).unsqueeze(1),
            (
                (matches_anchors[:, 1] - anchors[:, 1]) / (anchors[:, 3] * variances[0])
            ).unsqueeze(1),
            (torch.log(matches_anchors[:, 2] / anchors[:, 2]) / variances[1]).unsqueeze(
                1
            ),
            (torch.log(matches_anchors[:, 3] / anchors[:, 3]) / variances[1]).unsqueeze(
                1
            ),
        ),
        -1,
    )


def matching(anch,gt,labels,coords_space,labels_space,idx,variances=[0.1,0.2]):

    """
    anch are centered to corner anchors x1,y1,x2,y2  (with values in [0,1]) of dim [nb_of_anchors,4]
    gt are centered to corner anchors x1,y1,x2,y2 (with values in [0,1]) of dim [nb_gt_boxes,4]
    """

    # get  [nb_anchors,nb_gt_boxes] matrix of pairwise iou 
    intersection=iou(anch,gt)

    # for each gt best overlap with anchors 
    for_gt_id=intersection.argmax(dim=0)
    # for each anchor best overlap with gt 
    for_anchor_iou,for_anchor_idx, = intersection.max(dim=1)


    for i in range(gt.shape[0]):
        #for each gtbox look at assigned best anchor 
        best_assigned_anchor=for_gt_id[i]
        #for this best assigned anchor the associated gtbox has to be this gtbox at index i 
        for_anchor_idx[best_assigned_anchor]=i 
        #to make sure we dont drop this in next lines 
        for_anchor_iou[best_assigned_anchor]=1

    # we have just forced to make sure every gt box gets one anchor

    #class assign
    for_anchor_classes=labels[for_anchor_idx]
    #Background class is 0  (suppose FG  classes are from 1...K)
    for_anchor_classes[for_anchor_iou<0.5]=0
    #do same for coords
    for_anchor_coords=gt[for_anchor_idx]
    for_anchor_coords[for_anchor_iou<0.5]=0
    #encode 
    for_anchor_coords = corner_to_center(for_anchor_coords) 
    #note that for_anchor_coords are gt coords in center to corner version already 
    for_anchor_coords_encoded=encode(for_anchor_coords,anch,variances)

    
    coords_space[idx,:,:]=for_anchor_coords_encoded
    labels_space[idx,:,:]=for_anchor_classes.unsqueeze(1)