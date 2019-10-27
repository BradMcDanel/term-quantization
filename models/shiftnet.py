import torch
import torch.nn as nn
from .utils import load_state_dict_from_url

import sys 
sys.path.append('..')

from util import fuse
import booth
import shift

__all__ = ['ShiftNet', 'shiftnet19', 'convert_shiftnet19', 'convert_value_shiftnet19']

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

def convert_shiftnet19(model, w_sfs, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                       d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                       data_stationary):
        layers = []
        curr_layer = 0
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                layer = fuse(layer, model.features[i+1])
                if curr_layer < data_stationary: 
                    # ignore first layer (usually smaller than group size)
                    if layer.weight.shape[1] > 3:
                        layer.weight.data = booth.booth_cuda.radix_2_mod(layer.weight.data, w_sfs[curr_layer],
                                                                         w_stat_group, w_stat_terms)
                else:
                    layer.weight.data = booth.booth_cuda.radix_2_mod(layer.weight.data, w_sfs[curr_layer],
                                                                     w_move_group, w_move_terms)
            elif isinstance(layer, nn.ReLU):
                if i == len(model.features) - 1:
                    pass
                elif curr_layer < data_stationary:
                    layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.Radix2ModGroup(2**-7, d_move_group, d_move_terms),
                        )
                else:
                    layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.Radix2ModGroup(2**-7, d_stat_group, d_stat_terms),
                        )
 
            if not isinstance(layer, nn.BatchNorm2d):
                layers.append(layer)

            if isinstance(layer, nn.Conv2d):
                curr_layer += 1
        model.features = nn.Sequential(*layers)

        return model

def convert_value_shiftnet19(model, w_move_terms, w_move_group, w_stat_values, w_stat_group,
                             d_move_terms, d_move_group, d_stat_values, d_stat_group,
                             data_stationary):
        layers = []
        curr_layer = 0
        for i, layer in enumerate(model.features):
            if isinstance(layer, nn.Conv2d):
                layer = fuse(layer, model.features[i+1])
                if curr_layer < data_stationary: 
                    # ignore first layer (usually smaller than group size)
                    if layer.weight.shape[1] > 3:
                        layer.weight.data = booth.booth_cuda.value_group(layer.weight.data,
                                                                         w_stat_group, w_stat_values)
                else:
                    layer.weight.data = booth.booth_cuda.radix_2_mod(layer.weight.data, 2**-15,
                                                                     w_move_group, w_move_terms)
            elif isinstance(layer, nn.ReLU):
                if i == len(model.features) - 1:
                    pass
                elif curr_layer < data_stationary:
                    layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.Radix2ModGroup(2**-15, d_move_group, d_move_terms),
                        )
                else:
                    layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.ValueGroup(d_stat_group, d_stat_values),
                        )
 
            if not isinstance(layer, nn.BatchNorm2d):
                layers.append(layer)

            if isinstance(layer, nn.Conv2d):
                curr_layer += 1
        model.features = nn.Sequential(*layers)

        return model