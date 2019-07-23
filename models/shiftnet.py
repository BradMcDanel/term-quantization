import torch
import torch.nn as nn

import sys 
sys.path.append('..')

from cgm import CGM
from masked_conv import MaskedConv2d
import shift

__all__ = ['ShiftNet', 'ShiftNetCGM', 'shiftnet19', 'shiftnet19_cgm']

def shift_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return [
        shift.Shift(in_channels, kernel_size),
        nn.Conv2d(in_channels, out_channels, 1, stride, 0),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU()
    ]

def layer(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU()
    ]

def shift_layer_cgm(in_channels, out_channels, kernel_size=3, stride=1, cgm_groups=1):
    return [
        shift.Shift(in_channels, kernel_size),
        nn.Conv2d(in_channels, out_channels, 1, stride, 0),
        nn.BatchNorm2d(out_channels, affine=True),
        CGM(cgm_groups)
    ]

def layer_cgm(in_channels, out_channels, kernel_size=3, stride=2, padding=1, cgm_groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, affine=True),
        CGM(cgm_groups)
    ]


class ShiftNet(nn.Module):
    def __init__(self, settings, in_channels=3, num_classes=1000):
        super(ShiftNet, self).__init__()
        layers = []
        out_channels = settings[0][0]
        layers.extend(layer(in_channels, out_channels))
        in_channels = out_channels

        for out_channels, stride in settings:
            layers.extend(shift_layer(in_channels, out_channels, stride=stride))
            in_channels = out_channels

        self.feature_outchannels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_outchannels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.feature_outchannels)
        x = self.classifier(x)
        return x

class ShiftNetCGM(nn.Module):
    def __init__(self, settings, in_channels=3, num_classes=1000):
        super(ShiftNetCGM, self).__init__()
        layers = []
        out_channels = settings[0][0]
        layers.extend(layer_cgm(in_channels, out_channels, cgm_groups=2))
        in_channels = out_channels

        for out_channels, stride, cgm_groups in settings:
            layers.extend(shift_layer_cgm(in_channels, out_channels, stride=stride,
                                          cgm_groups=cgm_groups))
            in_channels = out_channels

        self.feature_outchannels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_outchannels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.feature_outchannels)
        x = self.classifier(x)
        return x

def shiftnet19(pretrained=False):
    settings = [
        [64, 1],
        [64, 1],
        [128, 2],
        [128, 1],
        [128, 1],
        [128, 1],
        [256, 2],
        [256, 1],
        [256, 1],
        [256, 1],
        [512, 2],
        [512, 1],
        [512, 1],
        [512, 1],
        [1024, 2],
        [1024, 1],
        [1024, 1],
    ]

    return ShiftNet(settings)


def shiftnet19_cgm(pretrained=False):
    settings = [
        [64, 1, 2],
        [64, 1, 2],
        [128, 2, 2],
        [128, 1, 2],
        [128, 1, 2],
        [128, 1, 2],
        [256, 2, 2],
        [256, 1, 2],
        [256, 1, 2],
        [256, 1, 2],
        [512, 2, 3],
        [512, 1, 3],
        [512, 1, 3],
        [512, 1, 3],
        [1024, 2, 3],
        [1024, 1, 3],
        [1024, 1, 2],
    ]

    return ShiftNetCGM(settings)