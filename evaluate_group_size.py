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

if __name__=='__main__':
    args = parser.parse_args()
    model = models.__dict__[args.arch](pretrained=True)
    model.cuda()
    train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
                                                                num_train=1000, num_val=50000)

    results = {}
    avg_terms = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    group_sizes = [1, 4, 8, 16]
    for group_size in group_sizes:
        name = str(group_size)
        results[name] = {'avg_terms': [], 'acc': []}
        for avg_term in avg_terms:
            if group_size == 1 and not avg_term.is_integer():
                continue
            term = int(group_size * avg_term)
            qmodel = models.convert_model(model, term, group_size, term, group_size,
                                          term, group_size, term, group_size, 100)
            criterion = nn.CrossEntropyLoss().cuda()
            qmodel = torch.nn.DataParallel(qmodel).cuda()
            _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=False)
            acc = acc.item()
            results[name]['avg_terms'].append(term / group_size)
            results[name]['acc'].append(acc)
            print(group_size, term, acc, term / group_size)

    with open('data/{}-group-results.txt'.format(args.arch), 'w') as fp:
        json.dump(results, fp)