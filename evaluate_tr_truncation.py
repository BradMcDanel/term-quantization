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

def get_weight_term_count(model, group_size, quant_func, minlength=None):
    if quant_func == 'hese':
        func = booth.num_hese_terms
    else:
        func = booth.num_binary_terms
    # w = model.features[48][1].weight.data
    # F, C = w.shape[0], w.shape[1]

    # w = w.view(F, C//group_size, group_size)
    # w_terms = func(w, w_sfs[12]).sum(2)
    # w_terms = np.array(w_terms.cpu().view(-1).tolist()).astype(int)

    x = model.features[24][0].x
    B, C, W, H = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
    x_terms = func(x, 2**-7)
    x_terms = x_terms.view(B*W*H, C//group_size, group_size).sum(2)
    x_terms = np.array(x_terms.cpu().view(-1).tolist()).astype(int)

    if minlength is None:
        return np.bincount(x_terms)
    else:
        return np.bincount(x_terms, minlength=minlength)

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
                                                                num_train=128, num_val=10)

    group_size = 8
    stat_terms = [8, 8]
    terms = [1000, 20]
    quant_funcs = ['binary', 'binary']

    bcs = []
    for i, (term, stat_term, quant_func) in enumerate(zip(terms, stat_terms, quant_funcs)):
        qmodel = copy.deepcopy(model).cpu()
        w_sfs = util.quantize_layers(qmodel, bits=8)
        qmodel.cuda()
        qmodel = models.convert_model(qmodel, w_sfs, stat_term, 1, term, group_size, stat_term,
                                      1, term, group_size, data_stationary=1,
                                      fuse_bn=False, quant_func=quant_func)

        criterion = nn.CrossEntropyLoss().cuda()
        util.add_average_trackers(qmodel, nn.Conv2d)
        qmodel.cuda()
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=True)

        bc = get_weight_term_count(qmodel, group_size, quant_func)
        bcs.append(bc)
    

    fill_colors = ['cornflowerblue', 'tomato']
    names = ['Original', 'Term Revealing']
    fig, axes = plt.subplots(2, sharex=True, sharey=True)
    max_x = len(bcs[0])
    max_y = max(100. * (bcs[1] / bcs[1].sum()))
    for i, (ax, bc, fill_color, name) in \
        enumerate(zip(axes, bcs, fill_colors, names)):
        tr = terms[1]
        xs = np.arange(len(bc))
        bc = 100. * (bc / bc.sum())
        ax.plot(xs, bc, '-', color='k', linewidth=1.0)
        ax.fill_between(xs, bc, color=fill_color, edgecolor=fill_color)
        if i == 0:
            ax.fill_between(xs[tr:], bc[tr:], facecolor="none", edgecolor='k', hatch='X', linewidth=0)

        ax.text(max_x + 0.25, max_y - 0.75, name, fontsize=16, ha='right')
        if i == 0:
            ax.axvline(tr, 0.046, bc[tr] / max_y + 0.27, color='r')
            ax.text(tr + 0.5,  bc[tr] + 0.75, "Term Revealing\nCut Point", fontsize=12, color='red')

    fig.text(0.02, 0.5, 'Frequency (%)', rotation=90, ha='center', va='center', fontsize=18)
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    axes[0].set_title('Data Term Frequency per Group of 8')
    axes[1].set_xlabel('Number of Data Terms')
    plt.savefig('figures/shiftnet-terms.png', dpi=300, bbox_inches='tight')
    plt.clf()
