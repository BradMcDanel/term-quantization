import argparse
import json
from copy import deepcopy

import torch
import torch.nn as nn

import cnn_models
import tr_layer
import profile_model
import util

def compute_avg_terms(tr_params):
    alphas = []
    for weight_bits, group_size, weight_terms in tr_params[1:]:
        alphas.append(weight_terms / group_size)

    return sum(alphas) / len(alphas)

def eval_model(args, model, weight_bits, group_size, weight_terms, data_bits,
               data_terms):
    # replace Conv2d with TRConv2dLayer
    tr_params = cnn_models.static_conv_layer_settings(model, weight_bits,
                                                      group_size, weight_terms)
    avg_terms = compute_avg_terms(tr_params)
    qmodel = cnn_models.convert_model(model, tr_params, data_bits, data_terms)

    # profiling
    qmc = deepcopy(qmodel)
    qmc.cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    tmacs, params = profile_model.get_model_ops(qmc, inputs=(x,))

    qmodel = nn.DataParallel(qmodel)

    # compute activation scale factors
    _ = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose, pct=0.05)
    tr_layer.set_tr_tracking(qmodel, False)

    # evaluate model performance
    _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose)

    return acc, tmacs, avg_terms, params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('val_dir', help='path to validation data folder')
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
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose flag')

    args = parser.parse_args()
    val_loader = util.get_imagenet_validation(args)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    model = cnn_models.__dict__[args.arch](pretrained=True).cuda(args.gpu)

    results = {}

    # Term Revealing Settings
    weight_bits = 9
    group_size = 8
    data_bits = 9
    data_terms = 3
    avg_term_settings = [1.0, 1.25, 1.5, 2.0, 3.0]
    group_sizes = [1, 2, 8, 16, 32]
    for group_size in group_sizes:
        key = str(group_size)
        results[key] = {'avg_terms': [], 'accs': [], 'tmacs': []}
        for avg_term in avg_term_settings:
            weight_terms = round(avg_term * group_size)
            res = eval_model(args, model, weight_bits, group_size,
                             weight_terms, data_bits, data_terms)
            acc, tmacs, avg_term, params = res
            print(data_terms, weight_terms, tmacs, acc)
            results[key]['accs'].append(acc)
            results[key]['tmacs'].append(tmacs)
            results[key]['avg_terms'].append(avg_term)

    with open('results/{}-group-size-results.json'.format(args.arch), 'w') as fp:
        json.dump(results, fp)