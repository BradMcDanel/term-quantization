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
import matplotlib.lines as lines
import matplotlib


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
                                                                num_train=128, num_val=12)

    group_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

    qmodel = copy.deepcopy(model).cpu()
    w_sfs = util.quantize_layers(qmodel, bits=8)
    qmodel.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    util.add_average_trackers(qmodel, nn.Conv2d)
    qmodel.cuda()
    _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=True)

    w = qmodel.features[24][1].weight.data
    F, C = w.shape[0], w.shape[1]

    x = qmodel.features[24][0].x
    B, C, W, H = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
    x_bin = np.array(torch.round(x / 2**-7 + 0.5).view(-1).tolist()).astype(int)
    x_terms = booth.num_binary_terms(x, 2**-7)

    means = []
    variances = []
    mins = []
    maxs = []
    for group_size in group_sizes:
        xt = x_terms.view(B*W*H, C//group_size, group_size).sum(2)
        xt = np.array(xt.cpu().view(-1).tolist()).astype(int)
        xt = xt / group_size
        means.append(np.mean(xt))
        variances.append(np.var(xt))
        mins.append(np.percentile(xt, 2.5))
        maxs.append(np.percentile(xt, 97.5))
    means = np.array(means)
    variances = np.array(variances)
    mins = np.array(mins)
    maxs = np.array(maxs)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.fill_between(group_sizes, mins, maxs, color='blue',
                    alpha=0.3, edgecolor=None, linewidth=0)
    ax.plot(group_sizes, mins, 'k-', linewidth=1.5)
    ax.plot(group_sizes, maxs, 'k-', linewidth=1.5)
    ax.plot(group_sizes, means, 'b-', linewidth=1.5)
    ax.text(8, means[3]+0.07, 'Mean (1.66)', fontsize=16)
    ax.text(8, maxs[3]+0.10, '97.5 percentile', fontsize=16, rotation=-14)
    ax.text(8, mins[3]+0.30, '2.5 percentile', fontsize=16, rotation=14)

    ax.set_xscale('log')
    ax.set_xticks(group_sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_tick_params(which='minor', size=0)
    ax.get_xaxis().set_tick_params(which='minor', width=0) 
    ax.set_title('(b) Group Term Variance')
    # plt.figtext(.5, 1,'(b) Larger Groups Reduce Variance', fontsize=15, ha='center')
    ax.set_xlabel('Group Size')
    ax.set_ylabel(r'Average # of Terms')
    plt.tight_layout()
    plt.savefig('figures/term-variance.pdf', bbox_inches='tight', dpi=300)
    plt.clf()