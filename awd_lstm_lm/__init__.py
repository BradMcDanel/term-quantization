from copy import deepcopy

import torch
import torch.nn as nn

from tr_layer import TRLSTMLayer

def replace_lstm_layers(model, tr_params, data_bits, data_terms):
    curr_layer = 0
    for name, layer in model.named_modules():
        if is_lstm_layer(layer):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            weight_bits, group_size, weight_terms = tr_params[curr_layer]
            layer = TRLSTMLayer(layer, data_bits, data_terms, weight_bits,
                                group_size, weight_terms)

            module._modules[module_keys[-1]] = layer
            curr_layer += 1

    return model

def static_lstm_layer_settings(model, weight_bits, group_size, num_terms):
    curr_layer = 0
    stats = []
    for _, layer in model.named_modules():
        if is_lstm_layer(layer):
            # if curr_layer == 0 or layer.groups > 1 or 'se' in name:
            #     stats.append((16, 1, 16))
            #     curr_layer += 1
            #     continue

            stats.append((weight_bits, group_size, num_terms))
            curr_layer += 1

    return stats

def is_lstm_layer(layer):
    return isinstance(layer, nn.LSTM)

def convert_model(model, tr_params, data_bits, data_terms):
    # copy the model, since we modify it internally
    model = deepcopy(model)
    return replace_lstm_layers(model, tr_params, data_bits, data_terms)
