from model.priorbox import AnchorBoxes
import yaml
import torch
import torch.nn as nn
import torch.nn.init as init
from model.detection import Detection


class SSD(nn.Module):
    def __init__(
        self,
        backbone,
        c4_name,
        priorbox_config,
        nb_classes,
        phase,
        prob_thr,
        nms_thr,
        top_k,
        variances,
        device,
        N_epochs: int = 100,
        alpha=1,
        c4_norm=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.nb_classes = nb_classes
        self.alpha = alpha
        self.N_epochs = N_epochs

        self.phase = phase
        self.prob_thr = prob_thr
        self.nms_thr = nms_thr
        self.top_k = top_k
        self.variances = variances
        self.device=device
        self.c4_norm = c4_norm

        self._hooked_features = []

        for name, module in backbone.named_modules():
            if name == c4_name:
                module.register_forward_hook(self._hook_fn)
                break

        # define extras layers
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

        boxes = AnchorBoxes(priorbox_config)
        anchors = boxes.forward().to(device)
        self.register_buffer("anchors", anchors)
        self.detection = Detection(
                nb_classes=nb_classes,
                prob_thr=prob_thr,
                nms_thr=nms_thr,
                top_k=top_k,
                variances=variances,
                anchors=self.anchors,
        )

    def _hook_fn(self, module, input, output):
        self._hooked_features.append(output)

    def _compute_locs_confs(self, X):
        """Shared feature extraction used by forward() and ONNX export."""
        self._hooked_features = []
        layers_for_prediction = []

        c5 = self.backbone(X)
        c4 = self._hooked_features[0]
        if self.c4_norm is not None:
            c4 = self.c4_norm(c4)
        layers_for_prediction = [c4]
        X = c5

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

        locs = loc.view(loc.size(0), -1, 4)
        # 8732 anchor boxes are sum of all anchor boxes across all ft map k =  of sum over k (Hk*Hk*ak)
        # for standard ssd300 it is 38*38*4+19*19*6+100*6+25*6+9*4+4
        confs = conf.view(conf.size(0), -1, self.nb_classes)
        return locs, confs

    def forward(self, X):
        locs, confs = self._compute_locs_confs(X)

        if self.phase == "train":
            return locs, confs
        elif self.phase == "test":
            output = self.detection(confs, locs)
            return locs, confs, output
        else:
            raise ValueError("Unknown phase. Expected train or test ")


class SSDOnnxWrapper(nn.Module):
    """Full inference graph for ONNX export: SSD features + Detection (NMS)."""

    def __init__(self, ssd: SSD):
        super().__init__()
        self.ssd = ssd

    def forward(self, x):
        locs, confs = self.ssd._compute_locs_confs(x)
        return self.ssd.detection(confs, locs)


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, padding, stride=1):
        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(input_channel, input_channel, kernel_size, groups=input_channel,
                      bias=False, padding=padding, stride=stride),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(input_channel, output_channel, 1, bias=False),
            nn.BatchNorm2d(output_channel),
        )

    def forward(self, x):
        return self.operation(x)


class DepthwiseSeparableExtraBlock(nn.Module):
    """Extra feature block following the torchvision SSDLite pattern:
    1x1 pointwise → 3x3 depthwise → 1x1 pointwise"""
    def __init__(self, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        mid_channels = out_channels // 2
        self.operation = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=padding,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.operation(x)

class SSDLite(SSD):

    def __init__(self, backbone_config_path, *args, **kwargs):
        with open(backbone_config_path, "r") as f:
            self._cfg = yaml.safe_load(f)
        #define priorbox config
        kwargs["priorbox_config"] = self._cfg
        super().__init__(*args, **kwargs)

        extras_cfg = self._cfg["extras"]
        anchors = self._cfg["anchors_per_location"]
        c4_ch = self._cfg["c4_channels"]

        self.extras = nn.ModuleList([
            DepthwiseSeparableExtraBlock(in_ch, out_ch, stride=s, padding=p)
            for in_ch, out_ch, s, p in extras_cfg
        ])
        self.extras.apply(weights_init)

        head_channels = [c4_ch] + [out_ch for _, out_ch, _, _ in extras_cfg]

        self.classification_convolutions = nn.ModuleList([
            DepthwiseSeparableConv(ch, a * self.nb_classes, kernel_size=3, padding=1)
            for ch, a in zip(head_channels, anchors)
        ])
        self.classification_convolutions.apply(weights_init)

        self.regression_convolutions = nn.ModuleList([
            DepthwiseSeparableConv(ch, a * 4, kernel_size=3, padding=1)
            for ch, a in zip(head_channels, anchors)
        ])
        self.regression_convolutions.apply(weights_init)




