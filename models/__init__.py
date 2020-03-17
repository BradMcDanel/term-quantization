from copy import deepcopy

import torch.nn as nn

from .alexnet import *
from .densenet import *
from .mobilenet import *
from .resnet import *
from .shiftnet import *
from .vgg import *
from .googlenet import *
import util
from tr_layer import TRConv2dLayer, profile_tensor


def replace_conv_layers(model, tr_params, data_bits):
    curr_layer = 0
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if curr_layer == 0:
                curr_layer += 1
                continue
 
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]
            
            weight_bits, group_size, num_terms = tr_params[curr_layer]
            module._modules[module_keys[-1]] = TRConv2dLayer(layer, data_bits, weight_bits,
                                                             group_size, num_terms)
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
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if curr_layer == 0 or layer.groups > 1:
                stats.append((16, 1, 16))
                curr_layer += 1
                continue
 
            stats.append((weight_bits, group_size, num_terms))
            curr_layer += 1

    return stats


def profile_conv_layers(model, err_tol):
    curr_layer = 0
    stats = []
    for _, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d):
            if curr_layer == 0 or layer.groups > 1:
                stats.append((16, 1, 16))
                curr_layer += 1
                continue
 
            stats.append(profile_tensor(layer.weight, err_tol))
            curr_layer += 1

    return stats


def convert_model(model, tr_params, data_bits, fuse_bn=False, quant_func='hese'):
    # copy the model, since we modify it internally
    model = deepcopy(model)
    return replace_conv_layers(model, tr_params, data_bits)


def convert_binary_model(model, w_move_terms, w_move_group, w_stat_terms, w_stat_group,
                         d_move_terms, d_move_group, d_stat_terms, d_stat_group,
                         data_stationary):

    # copy the model, since we modify it internally
    model = deepcopy(model)

    if isinstance(model, AlexNet):
        return convert_binary_alexnet(model, w_move_terms, w_move_group, w_stat_terms,
                                      w_stat_group, d_move_terms, d_move_group,
                                      d_stat_terms, d_stat_group, data_stationary)
    raise KeyError('Model: {} not found.', model.__class__.__name__)


def convert_value_model(model, w_move_terms, w_move_group, w_stat_values, w_stat_group,
                        d_move_terms, d_move_group, d_stat_values, d_stat_group,
                        data_stationary, encode_func):
    # copy the model, since we modify it internally
    model = deepcopy(model)

    if isinstance(model, ShiftNet):
        return convert_value_shiftnet19(model, w_move_terms, w_move_group, w_stat_values,
                                        w_stat_group, d_move_terms, d_move_group,
                                        d_stat_values, d_stat_group, data_stationary)
    elif isinstance(model, AlexNet):
        return convert_value_alexnet(model, w_move_terms, w_move_group, w_stat_values,
                                     w_stat_group, d_move_terms, d_move_group,
                                     d_stat_values, d_stat_group, data_stationary)
    elif isinstance(model, VGG):
        return convert_value_vgg(model, w_move_terms, w_move_group, w_stat_values,
                                 w_stat_group, d_move_terms, d_move_group,
                                 d_stat_values, d_stat_group, data_stationary)
    elif isinstance(model, ResNet):
        return ConvertedValueResNet(model, w_move_terms, w_move_group, w_stat_values,
                                    w_stat_group, d_move_terms, d_move_group,
                                    d_stat_values, d_stat_group, data_stationary)


    raise KeyError('Model: {} not found.', model.__class__.__name__)

def data_stationary_point(model):
    if isinstance(model, ShiftNet):
        return 12
    elif isinstance(model, AlexNet):
        return 6
    elif isinstance(model, VGG):
        return 12
    elif isinstance(model, ResNet):
        return 16

    raise KeyError('Model: {} not found.', model.__class__.__name__)