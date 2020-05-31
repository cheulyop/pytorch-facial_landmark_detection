import torch
import torch.nn as nn


# full convolution layer
def conv_bn(in_c, out_c, s):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=s, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


# depthwise separable convolution layer
def conv_dw(in_c, out_c, s):
    return nn.Sequential(
        # depthwise convolution applying single filter to each input channel
        # at groups=in_channels, each input channel is convolved with its own set of filters, of size out_c/in_c
        nn.Conv2d(in_c, in_c, 3, stride=s, padding=1, groups=in_c, bias=False),
        nn.BatchNorm2d(in_c),
        nn.ReLU(inplace=True),

        # pointwise convolution applying 1x1 convolution to combine outputs of depthwise convolution
        nn.Conv2d(in_c, out_c, 1, stride=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()

        # (in_c, out_c, s, n); n = number of repeat
        configurations = [
            (32, 64, 1, 1),
            (64, 128, 2, 1),
            (128, 128, 1, 1),
            (128, 256, 2, 1),
            (256, 256, 1, 1),
            (256, 512, 2, 1),
            (512, 512, 1, 5),
            (512, 1024, 2, 1),
            (1024, 1024, 1, 1),
        ]
        
        # only the first layer is a full convolution layer
        layers = [conv_bn(1, 32, 2)]
        for in_c, out_c, s, n in configurations:
            for _ in range(n):
                layers.append(conv_dw(in_c, out_c, s))
        
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 136)
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        out = self.fc(x).squeeze()

        return out