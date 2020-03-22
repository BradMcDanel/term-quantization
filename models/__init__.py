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

# Note: this is a forked/local version of thop to count ops for parent modules
import thop

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

def set_tr_tracking(model, tracking):
    for name, layer in model.named_modules():
        if isinstance(layer, TRConv2dLayer):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            module._modules[module_keys[-1]].tracking(tracking)

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

def get_model_ops(model):
    def tr_ops(m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.conv.in_channels // m.conv.groups * kernel_ops)

        # convert to term ops
        if m.group_size == 1:
            weight_terms = min(m.num_terms, m.weight_bits)
        else:
            weight_terms = m.num_terms
        data_terms = min(m.data_terms, m.data_bits)
        alpha = weight_terms / m.group_size
        total_ops = data_terms * alpha * total_ops

        if x.shape[1] > 3 and m.conv.groups == 1:
            m.conv.total_ops += torch.Tensor([int(total_ops)])

    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    custom_ops = {
        TRConv2dLayer: tr_ops,
        nn.Conv2d: thop.count_hooks.zero_ops,
        Conv2dStaticSamePadding: thop.count_hooks.zero_ops,
        nn.BatchNorm2d: thop.count_hooks.zero_ops,
        nn.Linear: thop.count_hooks.zero_ops,
        nn.AvgPool2d: thop.count_hooks.zero_ops,
        nn.AdaptiveAvgPool2d: thop.count_hooks.zero_ops
    }

    return thop.profile(model, inputs=(dummy_input,), custom_ops=custom_ops)

def convert_model(model, tr_params, data_bits, data_terms):
    # copy the model, since we modify it internally
    model = deepcopy(model)
    return replace_conv_layers(model, tr_params, data_bits, data_terms)
