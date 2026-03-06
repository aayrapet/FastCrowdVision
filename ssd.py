from priorbox import AnchorBoxes
import yaml
import torch
import torch.nn as nn
from l2norm import L2norm
import torch.nn.init as init
from detection import Detection

with open("config/priorbox.yaml", "r") as file:
    config = yaml.safe_load(file)


class SSD(nn.Module):
    def __init__(
        self,
        base,
        nb_classes,
        phase,
        prob_thr,
        nms_thr,
        top_k,
        variances,
        N_epochs: int = 100,

        alpha=1,
    ):
        super().__init__()
        self.features = base
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.N_epochs = N_epochs

        self.phase = phase
        self.prob_thr = prob_thr
        self.nms_thr = nms_thr
        self.top_k = top_k
        self.variances = variances

        self.l2norm = L2norm(512, 20)

        self.extras = nn.ModuleList(
            [
                # conv6 and conv7
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=512,
                        out_channels=1024,
                        kernel_size=3,
                        padding=6,
                        dilation=6,
                    ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                    nn.ReLU(inplace=True),
                ),
                # conv8_2
                nn.Sequential(
                    nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=256,
                        out_channels=512,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                    ),
                    nn.ReLU(inplace=True),
                ),
                # conv9_2
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        in_channels=128,
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                    ),
                    nn.ReLU(inplace=True),
                ),
                # conv10_2
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
                # conv11_2
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.extras.apply(weights_init)

        # define kernels for classification to output Feature Map of sieze H,W,ki*nb_classes for all H,W, ki in {4,6} for all i =1 ... |ssd feature maps | , ki is number of anchors for each location of H,W, image
        self.classification_convolutions = nn.ModuleList(
            [
                nn.Conv2d(512, 4 * nb_classes, kernel_size=3, padding=1),
                nn.Conv2d(1024, 6 * nb_classes, kernel_size=3, padding=1),
                nn.Conv2d(512, 6 * nb_classes, kernel_size=3, padding=1),
                nn.Conv2d(256, 6 * nb_classes, kernel_size=3, padding=1),
                nn.Conv2d(256, 4 * nb_classes, kernel_size=3, padding=1),
                nn.Conv2d(256, 4 * nb_classes, kernel_size=3, padding=1),
            ]
        )

        self.classification_convolutions.apply(weights_init)

        # same but using 4 coordinates for each anchor
        self.regression_convolutions = nn.ModuleList(
            [
                nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
                nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
                nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
                nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
                nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
                nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            ]
        )

        self.regression_convolutions.apply(weights_init)

        boxes = AnchorBoxes(config)
        anchors = boxes.forward()
        self.register_buffer("anchors", anchors)
        convs = {}
        i = 1
        convid = 1
        for id, el in enumerate(base):

            if isinstance(el, nn.Conv2d):

                convs[id] = f"{i}_{convid}"
                convid = convid + 1
            if isinstance(el, nn.MaxPool2d):
                convid = 1
                i = i + 1
        self.convs = convs

        
        self.detection = Detection(
                nb_classes=nb_classes,
                prob_thr=prob_thr,
                nms_thr=nms_thr,
                top_k=top_k,
                variances=variances,
                anchors=self.anchors,
        )

    def forward(self, X):
        layers_for_prediction = []

        # base model
        for idx in range(len(self.features)):

            X = self.features[idx](X)

            if (idx-1) in self.convs and self.convs[idx-1] == "4_3":       
                layers_for_prediction.append(self.l2norm(X))

        for idx in range(len(self.extras)):
            X = self.extras[idx](X)

            layers_for_prediction.append(X)

        classifications = []
        for layer_for_predictions, classification_convolution in zip(
            layers_for_prediction, self.classification_convolutions
        ):

            x = classification_convolution(layer_for_predictions)
            # then we want to get for all i,j in H*H and all k in 1....K -> p1.....pC probabilities of C classes
            """

            mathematically : 

            anchors=6
            total=6*21
            classes=21
            N=10
            H=19
            x=torch.randn((N,H,H,total))
            x.view(N,H,H,anchors,int(total/anchors)).shape

            x.view(N,H,H,anchors,int(total/anchors)).view(N,H*H*anchors,classes).shape

            However, this iplementation is slower as need to track nb_anchors and do manual calculations 
            which will slow down the process, this is why we do more standard code (this comment is for self learning purpose)
            """
            classifications.append(x.permute(0, 2, 3, 1).contiguous())

        regressions = []
        for layer_for_predictions, regression_convolution in zip(
            layers_for_prediction, self.regression_convolutions
        ):
            x = regression_convolution(layer_for_predictions)
            regressions.append(x.permute(0, 2, 3, 1).contiguous())

        # this efficient code was taken from degroot/ssd.pytorch github and is equivalent to my code in comment

        loc = torch.cat([o.view(o.size(0), -1) for o in regressions], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in classifications], 1)

        locs = loc.view(
            loc.size(0), -1, 4
        )  # so we get for every image 2D matrix for classification and regression : 8732 anchor boxes and nb coords/classes
        # 8732 anchor boxes are sum of all anchor boxes across all ft map k =  of sum over k (Hk*Hk*ak)
        # for standard ssd300 it is 38*38*4+19*19*6+100*6+25*6+9*4+4
        confs = conf.view(conf.size(0), -1, self.nb_classes)

        if self.phase == "train":
            return locs, confs
        elif self.phase == "test":
            output = self.detection(confs, locs)
            return locs, confs,output


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
