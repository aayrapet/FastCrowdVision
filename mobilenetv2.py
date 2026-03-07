

import torch 
import torch.nn as nn

class DepthwiseSepConv(nn.Module):
    def __init__(self,input_channel,t,kernel_size,output_channel,stride,dilation=1):
        """
        https://arxiv.org/pdf/1801.04381
        https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://docs.pytorch.org/vision/main/generated/torchvision.ops.Conv2dNormActivation.html
        https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py
        """
        self.residual_conn= (input_channel==output_channel) and stride==1
        padding=(kernel_size - 1) // 2 * dilation

        super().__init__()
        self.operation=nn.Sequential(

        nn.Conv2d(input_channel,input_channel*t,1,bias=False,padding=0),#bias is false since we use batchnorm
        nn.BatchNorm2d(input_channel*t),
        nn.ReLU6(inplace=True),

        #depthzise convolution 
        nn.Conv2d(input_channel*t,input_channel*t,kernel_size,groups=input_channel*t,stride=stride,bias=False,padding=padding),
        nn.BatchNorm2d(input_channel*t),
        nn.ReLU6(inplace=True),

        nn.Conv2d(input_channel*t,output_channel,1,bias=False,padding=0),
        nn.BatchNorm2d(output_channel),
        )
        

    def forward(self,x):
        if self.residual_conn:
            return x+self.operation(x)
        return self.operation(x)
    
class BottleneckBlock(nn.Module):
    def __init__(self,n,input_channel,t,kernel_size,output_channel,stride):
        """
        https://arxiv.org/pdf/1801.04381
        """
        super().__init__()
        chain=[]
        for i in range(n):
            chain.append(DepthwiseSepConv(input_channel,t,kernel_size,output_channel,stride))
            input_channel=output_channel
            stride=1
        self.operation=nn.Sequential(*chain)

    def forward(self,x):
        return self.operation(x)


#batch norm after each layer 
#drop out 
class MobileNetV2(nn.Module):
    def __init__(self,dropout,num_classes):
        super().__init__()
        self.features=nn.Sequential(
            
                nn.Conv2d(3,32,3,stride=2,padding=(3 - 1) // 2 ,bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),

                BottleneckBlock(n=1,input_channel=32,t=1,kernel_size=3,output_channel=16,stride=1),
                BottleneckBlock(n=2,input_channel=16,t=6,kernel_size=3,output_channel=24,stride=2),
                BottleneckBlock(n=3,input_channel=24,t=6,kernel_size=3,output_channel=32,stride=2),
                BottleneckBlock(n=4,input_channel=32,t=6,kernel_size=3,output_channel=64,stride=2),
                BottleneckBlock(n=3,input_channel=64,t=6,kernel_size=3,output_channel=96,stride=1),
                BottleneckBlock(n=3,input_channel=96,t=6,kernel_size=3,output_channel=160,stride=2),
                BottleneckBlock(n=1,input_channel=160,t=6,kernel_size=3,output_channel=320,stride=1),


                nn.Conv2d(320,1280,1,padding=0,bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),

            
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )

        # weight initialization from pythorch code taken 

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

    def forward(self,x):
        x=self.features(x)
        #directly use adaptive avg pooling so that resolutions shrink automatically to 1*1
        x=nn.functional.adaptive_avg_pool2d(x,(1,1))
        #pass to MLP : N images * 1280 pixels
        x=x.reshape(x.shape[0],-1)
        output=self.classifier(x)
        return output

