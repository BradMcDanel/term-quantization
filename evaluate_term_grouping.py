import argparse
import math
import copy
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
import pickle
from pprint import pprint
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
from torch.utils.data.dataset import Dataset
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid, save_image
import numpy as np
import json

import booth
import models
import util

def compute_avg_terms(tr_params):
    alphas = []
    for weight_bits, group_size, weight_terms in tr_params[1:]:
        alphas.append(weight_terms / group_size)

    return sum(alphas) / len(alphas)

def eval_model(args, model, weight_bits, group_size, weight_terms, data_bits,
               data_terms):
    # replace Conv2d with TRConv2dLayer
    tr_params = models.static_conv_layer_settings(model, weight_bits,
                                                  group_size, weight_terms)
    avg_terms = compute_avg_terms(tr_params)
    qmodel = models.convert_model(model, tr_params, data_bits, data_terms)

    # compute activation scale factors
    # _ = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose, pct=0.05)
    models.set_tr_tracking(qmodel, False)

    # evaluate model performance
    # _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose)
    tmacs, params = models.get_model_ops(qmodel)

    return 0, tmacs, 0, 0

    return acc.item(), tmacs, avg_terms, params


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--msgpack-loader', dest='msgpack_loader', action='store_true',
                    help='use custom msgpack dataloader')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose flag')

if __name__=='__main__':
    args = parser.parse_args()
    # train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
    #                                                             num_train=2500, num_val=50000)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    model = models.__dict__[args.arch](pretrained=True).cuda(args.gpu)

    results = {
        'quant': {
            'accs': [],
            'tmacs': [],
            'avg_terms': [],
            'params': [],
        },
        'tr-data2': {
            'accs': [],
            'tmacs': [],
            'avg_terms': [],
            'params': [],
        },
        'tr-data3': {
            'accs': [],
            'tmacs': [],
            'avg_terms': [],
            'params': [],
        },
        'tr-data4': {
            'accs': [],
            'tmacs': [],
            'avg_terms': [],
            'params': [],
        }
    }

    # Traditional Quantization Settings
    weight_bits = 9
    group_size = 1
    weight_terms = 9
    data_bits = 9
    data_terms = 9
    weight_bit_settings = [6, 7, 8, 9]
    for weight_bits in weight_bit_settings:
        res = eval_model(args, model, weight_bits, group_size, weight_terms,
                         data_bits, data_terms)
        acc, tmacs, avg_terms, params = res
        # print(acc)
        print(weight_bits, tmacs)
        results['quant']['accs'].append(acc)
        results['quant']['tmacs'].append(tmacs)
        results['quant']['avg_terms'].append(avg_terms)
        results['quant']['params'].append(params)

    # Term Revealing Settings
    weight_bits = 9
    group_size = 8
    data_bits = 9
    data_term_settings = [2, 3, 4]
    weight_term_settings = [12, 16, 20, 24]
    for data_terms in data_term_settings:
        key = 'tr-data{}'.format(data_terms)
        for weight_terms in weight_term_settings:
            res = eval_model(args, model, weight_bits, group_size,
                             weight_terms, data_bits, data_terms)
            acc, tmacs, avg_terms, params = res
            print(data_terms, weight_terms, tmacs)
            results[key]['accs'].append(acc)
            results[key]['tmacs'].append(tmacs)
            results[key]['avg_terms'].append(avg_terms)
            results[key]['params'].append(params)

    assert False
    with open('data/{}-results.txt'.format(args.arch), 'w') as fp:
        json.dump(results, fp)