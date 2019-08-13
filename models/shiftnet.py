import torch
import torch.nn as nn
from .utils import load_state_dict_from_url

import sys 
sys.path.append('..')

import shift

__all__ = ['ShiftNet', 'shiftnet19']

model_urls = {
    'shiftnet19': 'https://github.com/BradMcDanel/term-grouping/blob/master/pth/shiftnet19-d372c4d2.pth?raw=true'
}

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

def shiftnet19(pretrained=False, progress=True):
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

    model = ShiftNet(settings)
    if pretrained:
        state_dict = torch.load('pth/shiftnet19-d372c4d2.pth')
        # state_dict = load_state_dict_from_url(model_urls['shiftnet19'],
        #                                       progress=progress)
        model.load_state_dict(state_dict)

    return model