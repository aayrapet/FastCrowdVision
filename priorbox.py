
from utils import calculate_anchor_w_h1,calculate_anchor_w_h2,calculate_sk
from utils import normalised_anchor_coords,corner_to_center_scalar
import math as mt
import torch 
class AnchorBoxes():
    def __init__(self,config):
        self.aspect_ratios=config["aspect_ratios"]
        self.feature_maps=config["feature_maps"]
    
    def __calculate_sk_per_ft_map(self):
        FT_MAP_ANCHOR_W_H={}
        M=len(self.feature_maps)
        for k in range(1,M+1):
            sk=calculate_sk(k,M)
            w_h=[]
            for a in self.aspect_ratios[k-1]:
                if a==1:
                    sk_plus_1=calculate_sk(k+1,M)
                    sk_prime=mt.sqrt(sk*sk_plus_1)
                    w_h.append(calculate_anchor_w_h1(sk_prime,a))
                    w_h.append(calculate_anchor_w_h1(sk,a))
                else:
                    w_h.append(calculate_anchor_w_h1(sk,a))
                    #revert logic
                    w_h.append(calculate_anchor_w_h2(sk,a))
            FT_MAP_ANCHOR_W_H[k]=w_h
        return FT_MAP_ANCHOR_W_H
            
    def forward(self):

        anchors=[]
        FT_MAP_ANCHOR_W_H=self.__calculate_sk_per_ft_map()

        for k,f in enumerate(self.feature_maps,start=1):

            for w_k_a,h_k_a in FT_MAP_ANCHOR_W_H[k]:
                for i in range(f):
                    for j in range(f):
                
                        ax1,ay1,ax2,ay2=normalised_anchor_coords(i,j,f,w_k_a,h_k_a)
                        cx, cy, w, h=corner_to_center_scalar(ax1,ay1,ax2,ay2)
                        anchors.append([cx, cy, w, h])

        anchors_tensor = torch.tensor(anchors, dtype=torch.float32)
        return anchors_tensor
    

if __name__=="__main__":
    import yaml

    with open('config/priorbox.yaml', 'r') as file:
        config = yaml.safe_load(file)

    boxes=AnchorBoxes(config)
    n=boxes.forward()
    print(n.shape)
    print(n)