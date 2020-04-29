from copy import deepcopy

import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import (
    alexnet,
    vgg16_bn,
    resnet18,
    mobilenet_v2
)
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

import util
from tr_layer import TRConv2dLayer
from .shiftnet import shiftnet19

def model_names():
    return ['alexnet', 'vgg16_bn', 'resnet18', 'efficientnet_b0', 'shiftnet19', 'mobilenet_v2']

def efficientnet_b0(pretrained=True):
    if pretrained:
        return EfficientNet.from_pretrained('efficientnet-b0')

    return EfficientNet.from_name('efficientnet-b0')

def is_conv_layer(layer):
    return isinstance(layer, (nn.Conv2d, Conv2dStaticSamePadding))

def replace_conv_layers(model, tr_params, data_bits, data_terms):
    curr_layer = 0
    for name, layer in model.named_modules():
        if is_conv_layer(layer):
            if curr_layer == 0:
                curr_layer += 1
                continue

            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            weight_bits, group_size, weight_terms = tr_params[curr_layer]
            layer = TRConv2dLayer(layer, data_bits, data_terms, weight_bits,
                                  group_size, weight_terms)

            module._modules[module_keys[-1]] = layer
            curr_layer += 1

    return model

def static_conv_layer_settings(model, weight_bits, group_size, num_terms):
    curr_layer = 0
    stats = []
    for name, layer in model.named_modules():
        if is_conv_layer(layer):
            if curr_layer == 0 or layer.groups > 1 or 'se' in name:
                stats.append((16, 1, 16))
                curr_layer += 1
                continue

            stats.append((weight_bits, group_size, num_terms))
            curr_layer += 1

    return stats

def convert_model(model, tr_params, data_bits, data_terms):
    # copy the model, since we modify it internally
    model = deepcopy(model)
    return replace_conv_layers(model, tr_params, data_bits, data_terms)
