import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import matching



def HNM_mine(classifications_reshaped,labels_reshaped):

    """
    Input: 

    reshaped classifications; shape [N_anhcores(8732),N_classes] 
    reshaped labels; shape [N_anhcores] , one vector containing classes 

    Output:

    all; shape [positive anchors + top negative anchors]

    For more details : https://arxiv.org/pdf/1512.02325, page 6 

    """
    losses=F.cross_entropy(classifications_reshaped,labels_reshaped,reduction="none") 

    negative_indexes=torch.nonzero(labels_reshaped==0,as_tuple=True,)[0]
    positive_indexes=torch.nonzero(labels_reshaped>0,as_tuple=True,)[0]
    nb_positives=positive_indexes.numel()


    _,indx=losses[negative_indexes].sort(descending=True)

    negative_indexes=negative_indexes[indx[:min(nb_positives*4,len(indx))]]
    
    all=torch.cat([negative_indexes, positive_indexes], dim=0)
    return all

def HNM_max(classifications, labels, neg_pos_ratio=4):

    """
    vectorised version of HNM_mine, for all images in a batch 
    classifications: [N, A, C]
    labels:          [N, A]
    returns:         flat indices into (N*A)
    """

    N, A, C = classifications.shape

    #compute per-anchor loss (keep [N, A])
    loss_c = F.cross_entropy(
        classifications.view(-1, C),
        labels.view(-1),
        reduction="none"
    ).view(N, A)


    pos = labels > 0
    loss_c[pos] = 0
    _, loss_idx = loss_c.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)
    num_pos = pos.sum(1, keepdim=True)

    num_neg = torch.clamp(neg_pos_ratio * num_pos, max=A - 1)

    neg = idx_rank < num_neg.expand_as(idx_rank)
    selected = (pos | neg)
    return selected.view(-1).nonzero(as_tuple=True)[0]


class MultiLoss(nn.Module):
    def __init__(self,anchors):
        super().__init__()
        self.anchors=anchors
        pass
    
    def forward(self,start_index,end_index,gt_list,labels_list,regressions,classifications):
        n_anchors=anchors.shape[0]

        N_images=end_index-start_index
        
        coords_space=torch.empty((N_images,n_anchors,4))
        labels_space=torch.empty((N_images,n_anchors,1)).long()

        #execute 2D matching for every image 
 
        for i in range(start_index,end_index):
            matching(self.anchors,gt_list[i],labels_list[i],coords_space,labels_space,i)
           
        labels_space = labels_space.squeeze(-1)  
        pos = labels_space > 0  
        nb_pos = pos.sum()
        
        if nb_pos == 0:
            # No positive samples in entire batch
            Loss_loc = torch.tensor(0., device=classifications.device)
            Loss_conf = torch.tensor(0., device=classifications.device)
            no_pos = True
            return Loss_loc, Loss_conf, no_pos

        Loss_loc=F.smooth_l1_loss(regressions[pos],coords_space[pos], reduction='sum')/(nb_pos)

        hnm_selected=HNM_max(classifications,## [N, A, C]
        labels_space# [N, A]

        )

        
        Loss_conf=F.cross_entropy(
                classifications.reshape(-1,classifications.size(-1))[hnm_selected],
                labels_space.reshape(-1)[hnm_selected],
                reduction="sum"
        ) / nb_pos
        no_pos=False
        
        
        return Loss_loc,Loss_conf,no_pos



