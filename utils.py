
import math as mt 
def iou(x1A,y1A,x2A,y2A,x1B,y1B,x2B,y2B):
    a=min(x2A,x2B)
    b=max(x1B,x1A)
    d=max(y1A,y1B)
    c=min(y2A,y2B)

    h=max(0,a-b)
    w=max(0,c-d)
    intersection=h*w

    A_area=(x2A-x1A)*(y2A-y1A)
    B_area=(x2B-x1B)*(y2B-y1B)

    union=A_area+B_area-intersection

    return intersection/union
def calculate_sk(k,m):
    """
    k is index of ft map , m is total nb of ft map 
    (ft map are map participating in prediction)
    k=1....M
    """
    smin,smax=0.2,0.9
    if k>m:
        return 1
    return smin+(smax-smin)*(k-1)/(m-1)


def calculate_anchor_w_h1(sk,a):
    return sk*mt.sqrt(a),sk/mt.sqrt(a)

def calculate_anchor_w_h2(sk,a):
    return sk/mt.sqrt(a),sk*mt.sqrt(a)


def normalised_anchor_coords(i,j,f,w,h):

    """
    i , j in len(feature map)=0...f-1
    w=w_k_a
    h=h_k_a
    """
    centerx=(j+0.5)/f
    centery=(i+0.5)/f


    x2=min(centerx+w/2,1)
    y2=min(centery+h/2,1)

    x1=max(centerx-w/2,0)
    y1=max(centery-h/2,0)
    return x1,y1,x2,y2


def normalised_gt_coords(x1,y1,x2,y2,H,W):
    """
    normalise gt coords so that they are in [0,1]
    """
    return x1/W,y1/H,x2/W,y2/H

def corner_to_center(x1, y1, x2, y2):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx, cy, w, h


def center_to_corner(cx, cy, w, h):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


