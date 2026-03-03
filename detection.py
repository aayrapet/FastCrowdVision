from utils import center_to_corner, decode
from torchvision.ops import nms
import torch
import torch.nn as nn
import torch.nn.functional as F

class Detection(nn.Module):
    """
    Performing Non-max Supression on model outputs 

    """
    def __init__(self, nb_classes, prob_thr, nms_thr, top_k, variances: list, anchors):
        super().__init__()
        self.prob_thr = prob_thr
        self.top_k = top_k
        self.variances = variances
        self.nb_classes = nb_classes
        self.anchors = anchors
        self.nms_thr = nms_thr

    def forward(self, classifications, regressions):
        N_images = classifications.shape[0]
        output = torch.zeros(N_images, self.nb_classes, self.top_k, 6, device=classifications.device)
        for i in range(N_images):
            image_class = classifications[i, :, :]
            image_regress = regressions[i, :, :]

            sft = F.softmax(image_class, dim=1)
            decoded = decode(image_regress, self.anchors, self.variances)

            # do nms detection only for non background classes
            for j in range(1, image_class.shape[1]):
                #select only > thr bboxes
                class_probas = sft[:, j]
                filter = class_probas > self.prob_thr
                selected_class_probas = class_probas[filter]
                if selected_class_probas.numel()==0:
                    continue
                selected_bbox = decoded[filter]
                selected_bbox = center_to_corner(selected_bbox)
                selected_bbox = selected_bbox.clamp(min=0, max=1)
                

                nms_idx = nms(selected_bbox, selected_class_probas, self.nms_thr)

                upper_bound=min(self.top_k,selected_bbox[nms_idx].shape[0])
                output[i, j, : upper_bound, :] = torch.cat(
                    (
                        #select top k bboxes per class and probas 
                        selected_bbox[nms_idx][: upper_bound, :],
                        selected_class_probas[nms_idx][: upper_bound].unsqueeze(1),
                        #indicate class label also 
                        torch.ones(
                            upper_bound, dtype=torch.float32, device=regressions.device
                        ).unsqueeze(1)
                        * j,
                    ),
                    -1,
                )

        #top k now over all image (among all classes select top k bboxes)
        output = output.reshape(N_images, self.nb_classes * self.top_k, 6)
        #based on 5th column which is probability 
        scores = output[:, :, -2]
       
        _, idx = scores.topk(self.top_k, dim=1)

        top = output.gather(
            dim=1, index=idx.unsqueeze(-1).expand(-1, -1, output.size(-1))
        )

        return top
