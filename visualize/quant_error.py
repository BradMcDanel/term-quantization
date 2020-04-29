import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cnn_models
import tr_layer

import util
import visualize
plt = visualize.import_plt_settings(local_display=True)
import matplotlib.patches as patches

import bit_utils

def get_layers(model, layer_types):
    layers = []
    for _, child in model.named_children():
        if isinstance(child, layer_types):
            layers.append(child)
        else:
            layers.extend(get_layers(child, layer_types))

    return layers


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Group Distribution Example')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                        choices=cnn_models.model_names(),
                        help='model architecture: ' +
                        ' | '.join(cnn_models.model_names()) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose flag')

    args = parser.parse_args()
    model = cnn_models.__dict__[args.arch](pretrained=True).cuda(args.gpu)
    weight_bits = 9
    group_size = 8
    data_bits = 9
    data_terms = 9
    weight_terms = 9
    group_size = 1

    orig_layers = get_layers(model, nn.Conv2d)
    # drop initial layer
    orig_layers.pop(0)

    # Quant settings
    total_errors = []
    weight_bit_settings = [9, 7, 6]
    for weight_bits in weight_bit_settings:
        errors = []
        tr_params = cnn_models.static_conv_layer_settings(model, weight_bits, group_size, weight_terms)
        qmodel = cnn_models.convert_model(model, tr_params, data_bits, data_terms)
        quant_layers = get_layers(qmodel, tr_layer.TRConv2dLayer)

        for i, (orig_layer, quant_layer) in enumerate(zip(orig_layers, quant_layers)):
            if i == 0: continue
            err = (orig_layer.weight - quant_layer.conv.weight).abs().mean()
            errors.append(err)

        total_errors.append(errors)

    num_term_settings = [14]
    weight_bits = 9
    group_size = 8
    for weight_terms in num_term_settings:
        errors = []
        tr_params = cnn_models.static_conv_layer_settings(model, weight_bits, group_size, weight_terms)
        qmodel = cnn_models.convert_model(model, tr_params, data_bits, data_terms)
        quant_layers = get_layers(qmodel, tr_layer.TRConv2dLayer)

        for i, (orig_layer, quant_layer) in enumerate(zip(orig_layers, quant_layers)):
            if i == 0: continue
            err = (orig_layer.weight - quant_layer.conv.weight).abs().mean()
            errors.append(err)

        total_errors.append(errors)


    plt.figure(figsize=(10, 3.5))
    names = ['8-bit QT', '7-bit QT', '6-bit QT',  'TR (g=8, k=14)']
    width = 0.15
    x_ticks = np.arange(len(total_errors[0]))
    for i, error in enumerate(total_errors):
        plt.bar(x_ticks + i*width, error, label=names[i], width=width, edgecolor='k')
    plt.legend(loc='upper center', ncol=4)
    plt.tight_layout()
    plt.title('Layerwise Quantization Error in ResNet-18')
    plt.ylim(ymax=0.0075)
    plt.ylabel('Error per Weight (avg)')
    plt.xlabel('ResNet-18 Layer')
    plt.xticks(x_ticks+(1.5*width), x_ticks+1)
    plt.savefig('figures/quant_error_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.clf()

