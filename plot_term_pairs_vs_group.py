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

def get_term_count(model, group_size, quant_func, minlength=None):
    if quant_func == 'hese':
        func = booth.num_hese_terms
    else:
        func = booth.num_binary_terms
    x = model.features[24][0].x
    w = model.features[24][1].weight.data
    B, C, W, H = x.shape
    F = w.shape[0]
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
    x_terms = func(x, 2**-7)
    x_terms = x_terms.view(B*W*H, C)
    w = w.view(F, C)
    w_terms = func(w, w_sfs[12])
    w_terms = w_terms.view(F, C)

    all_res = []
    for i in range(0, C//group_size):
        start, end = i*group_size, (i+1)*group_size
        res = torch.matmul(x_terms[:, start:end], w_terms[:, start:end].transpose(0, 1))
        all_res.extend(res.view(-1).tolist())

    all_res = np.array(all_res).astype(int)

    if minlength is None:
        return np.bincount(all_res)
    else:
        return np.bincount(all_res, minlength=minlength)

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
                                                                num_train=128, num_val=4)

    group_sizes = [1, 2, 4, 8, 16]

    bcs = []
    for group_size in group_sizes:
        print(group_size)
        qmodel = copy.deepcopy(model).cpu()
        w_sfs = util.quantize_layers(qmodel, bits=8)
        qmodel.cuda()

        criterion = nn.CrossEntropyLoss().cuda()
        util.add_average_trackers(qmodel, nn.Conv2d)
        qmodel.cuda()
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=True)

        bc = get_term_count(qmodel, group_size, 'binary')
        bcs.append(bc)
    

    fig, axes = plt.subplots(5, figsize=(5, 4.5), sharex=True)
    yticks = [(0, 25), (0, 12), (0, 3), (0, 2), (0, 1)]
    for i, (ax, bc, group_size, ytick) in enumerate(zip(axes, bcs, group_sizes, yticks)):
        bc = 100. * (bc / bc.sum())
        max_x = len(bcs[-1]) + 5
        max_y = max(bc) * 0.75
        long_tail = np.arange(len(bc))[np.cumsum(bc) > 99][0]
        last_x = len(bc)

        ax.fill_between(np.arange(len(bc)), bc)
        ax.plot(np.arange(len(bc)), bc, '-', color='k', linewidth=1.0)

        if i == 0:
            offset = 0
        elif i == 4:
            offset = -30
        else:
            offset = -10

        # tail
        ax.annotate('99%({})'.format(long_tail), xy=(long_tail, max_y*0.05),  textcoords='data',
                    xytext=(long_tail-10, max_y * 0.5), fontsize=12, arrowprops=dict(arrowstyle="-"))
        ax.plot([long_tail], [max_y*0.05], 'or', markeredgecolor='k', ms=4)

        # max
        ax.annotate('Max({})'.format(last_x), xy=(last_x, max_y*0.05),  textcoords='data',
                    xytext=(last_x+offset, max_y * 0.5), fontsize=12, arrowprops=dict(arrowstyle="-"))
        ax.plot([last_x], [max_y*0.05], 'or', markeredgecolor='k', ms=4)

        ax.set_yticks(ytick)
        ax.text(max_x, max_y, 'Group of {}'.format(group_size), fontsize=14, ha='right')

    fig.text(0.01, 0.5, 'Frequency of Term Pairs', rotation=90,
             ha='center', va='center', fontsize=18)
    axes[0].set_title('(a) Term Pairs per Group Size')
    axes[4].set_xlabel('Number of Term Pairs')
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.savefig('figures/term-group-dist.pdf', dpi=300, bbox_inches='tight')
    plt.clf()