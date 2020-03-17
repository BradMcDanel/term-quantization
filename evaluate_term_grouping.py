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
plt = util.import_plt_settings(local_display=False)

def compute_avg_terms(tr_params):
    alphas = []
    for weight_bits, group_size, num_terms in tr_params[1:]:
        alphas.append(num_terms / group_size)

    return sum(alphas) / len(alphas)


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
    train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
                                                                num_train=2500, num_val=50000)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    model = models.__dict__[args.arch](pretrained=True).cuda(args.gpu)

    results = {
        'static': {
            'avg_terms': [],
            'accs': []
        },
        'dynamic': {
            'avg_terms': [],
            'accs': []
        }
    }

    # Static settings
    static_terms = [16, 20, 24, 28, 32]
    for num_terms in static_terms:
        # replace Conv2d with TRConv2dLayer
        tr_params = models.static_conv_layer_settings(model, 10, 16, num_terms)
        # pprint(tr_params)
        avg_terms = compute_avg_terms(tr_params)
        qmodel = models.convert_model(model, tr_params, 9)

        # compute activation scale factors
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose, pct=0.05)
        models.set_tr_tracking(qmodel, False)

        # evaluate model performance
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose)
        print(avg_terms, acc.item())
        results['static']['avg_terms'].append(avg_terms)
        results['static']['accs'].append(acc.item())


    # Dynamic settings
    err_tols = [0.0032, 0.0028, 0.0024, 0.002, 0.0016, 0.0012, 0.0008, 0.0004]
    for err_tol in err_tols:
        # replace Conv2d with TRConv2dLayer
        tr_params = models.profile_conv_layers(model, err_tol=err_tol)
        # print(err_tol)
        # pprint(tr_params)
        avg_terms = compute_avg_terms(tr_params)
        qmodel = models.convert_model(model, tr_params, 9)

        # compute activation scale factors
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose, pct=0.05)
        models.set_tr_tracking(qmodel, False)

        # evaluate model performance
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose)
        print(avg_terms, acc.item())
        results['dynamic']['avg_terms'].append(avg_terms)
        results['dynamic']['accs'].append(acc.item())



    with open('data/{}-results.txt'.format(args.arch), 'w') as fp:
        json.dump(results, fp)