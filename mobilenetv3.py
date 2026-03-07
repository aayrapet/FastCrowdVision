import torch
import torch.nn as nn
from utils import _make_divisible

class SqueezeExcitationBlock(nn.Module):
    def __init__(self, input_size, factor):
        """
        expects 3D Tensor
        """
        super().__init__()
        self.operation = nn.Sequential(
            # squeeze
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            # excitation
            nn.Linear(input_size, _make_divisible(input_size // factor, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(input_size // factor, 8), input_size),
            nn.Hardsigmoid(inplace=True),
        )

    def forward(self, x):
        f = self.operation(x)
        f = f.view(f.shape[0], f.shape[1], 1, 1).contiguous()
        return x * f

def _get_activation(activation):
    if activation == "re":
        return nn.ReLU(inplace=True)
    elif activation == "hs":
        return nn.Hardswish(inplace=True)
    else:
        raise ValueError("not known activation")

class DepthwiseSepConv(nn.Module):
    def __init__(
        self,
        input_channel,
        expansion_size,
        kernel_size,
        output_channel,
        stride,
        dilation=1,
        SE: bool = False,
        activation="re",
    ):
        """
        https://arxiv.org/pdf/1801.04381
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://docs.pytorch.org/vision/main/generated/torchvision.ops.Conv2dNormActivation.html
        https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        """
      
        super().__init__()
        self.residual_conn = (input_channel == output_channel) and stride == 1
        padding = (kernel_size - 1) // 2 * dilation

        
        layers = [
            nn.Conv2d(
                input_channel, expansion_size, 1, bias=False, padding=0
            ),  # bias is false since we use batchnorm
            nn.BatchNorm2d(expansion_size),
            _get_activation(activation),
            # depthzise convolution
            nn.Conv2d(
                expansion_size,
                expansion_size,
                kernel_size,
                groups=expansion_size,
                stride=stride,
                bias=False,
                padding=padding,
            ),
            nn.BatchNorm2d(expansion_size),
            _get_activation(activation),
        ]
        if SE:
            layers.append(SqueezeExcitationBlock(expansion_size, 4))
        layers.extend([
            nn.Conv2d(expansion_size, output_channel, 1, bias=False, padding=0),
            nn.BatchNorm2d(output_channel),
        ] 
        )
        self.operation = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_conn:
            return x + self.operation(x)
        return self.operation(x)


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        n,
        input_channel: int,
        kernel_size: list[int],
        expansion_size: list[int],
        output_channel: list[int],
        stride: list[int],
        NL: list[str],
        SE: list[bool],
    ):

        super().__init__()
        chain = []
        for i in range(n):
            chain.append(
                DepthwiseSepConv(
                    input_channel,
                    expansion_size[i],
                    kernel_size[i],
                    output_channel[i],
                    stride[i],
                    SE=SE[i],
                    activation=NL[i],
                )
            )
            input_channel = output_channel[i]
        self.operation = nn.Sequential(*chain)

    def forward(self, x):
        return self.operation(x)


class MobileNetV3Large(nn.Module):
    def __init__(self, dropout, num_classes, last_channel):

        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            BottleneckBlock(
                n=2,
                input_channel=16,
                kernel_size=[3, 3],
                expansion_size=[16, 64],
                output_channel=[16, 24],
                SE=[False, False],
                stride=[1, 2],
                NL=["re", "re"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=24,
                kernel_size=[3, 5],
                expansion_size=[72, 72],
                output_channel=[24, 40],
                SE=[False, True],
                stride=[1, 2],
                NL=["re", "re"],
            ),
            BottleneckBlock(
                n=3,
                input_channel=40,
                kernel_size=[5, 5,3],
                expansion_size=[120, 120,240],
                output_channel=[40, 40,80],
                SE=[True, True,False],
                stride=[1, 1,2],
                NL=["re", "re","hs"],
            ),

            BottleneckBlock(
                n=4,
                input_channel=80,
                kernel_size=[3,3,3,3],
                expansion_size=[200, 184,184,480],
                output_channel=[80, 80,80,112],
                SE=[False, False,False,True],
                stride=[1, 1,1,1],
                NL=["hs", "hs","hs","hs"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=112,
                kernel_size=[3,5],
                expansion_size=[672,672],
                output_channel=[112,160],
                SE=[True,True],
                stride=[1, 2],
                NL=["hs", "hs"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=160,
                kernel_size=[5,5],
                expansion_size=[960,960],
                output_channel=[160,160],
                SE=[True,True],
                stride=[1, 1],
                NL=["hs", "hs"],
            ),
            nn.Conv2d(160,960,1,padding=0,stride=1,bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),

        )

        self.classifier = nn.Sequential(
            nn.Linear(960, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )
        #code taken from pytorch library 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x=self.operation(x)
        x=x.reshape(x.shape[0],-1)
        output=self.classifier(x)
        return output


class MobileNetV3Small(nn.Module):
    def __init__(self, dropout, num_classes, last_channel):

        super().__init__()
        self.operation = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            BottleneckBlock(
                n=1,
                input_channel=16,
                kernel_size=[3],
                expansion_size=[16],
                output_channel=[16],
                SE=[True],
                stride=[2],
                NL=["re"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=16,
                kernel_size=[3, 3],
                expansion_size=[72, 88],
                output_channel=[24, 24],
                SE=[False, False],
                stride=[2, 1],
                NL=["re", "re"],
            ),
            BottleneckBlock(
                n=4,
                input_channel=24,
                kernel_size=[5, 5, 5, 5],
                expansion_size=[96, 240, 240, 120],
                output_channel=[40, 40, 40, 48],
                SE=[True, True, True, True],
                stride=[2, 1, 1, 1],
                NL=["hs", "hs", "hs", "hs"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=48,
                kernel_size=[5, 5],
                expansion_size=[144, 288],
                output_channel=[48, 96],
                SE=[True, True],
                stride=[1, 2],
                NL=["hs", "hs"],
            ),
            BottleneckBlock(
                n=2,
                input_channel=96,
                kernel_size=[5, 5],
                expansion_size=[576, 576],
                output_channel=[96, 96],
                SE=[True, True],
                stride=[1, 1],
                NL=["hs", "hs"],
            ),
            nn.Conv2d(96, 576, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Linear(576, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.operation(x)
        x = x.reshape(x.shape[0], -1)
        output = self.classifier(x)
        return output

