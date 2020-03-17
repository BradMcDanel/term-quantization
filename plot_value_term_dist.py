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
                                                                num_train=128, num_val=4)

    group_size = 1

    qmodel = copy.deepcopy(model).cpu()
    w_sfs = util.quantize_layers(qmodel, bits=8)
    qmodel.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    util.add_average_trackers(qmodel, nn.Conv2d)
    qmodel.cuda()
    _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=True)

    w = qmodel.features[24][1].weight.data
    F, C = w.shape[0], w.shape[1]

    w = w.view(F, C//group_size, group_size)
    w_sfs[12] = 2**-9
    w_bin = np.array(torch.round(w / w_sfs[12] + 0.5).view(-1).tolist()).astype(int)
    w_terms = booth.num_binary_terms(w, w_sfs[12]).sum(2)
    w_terms = np.array(w_terms.cpu().view(-1).tolist()).astype(int)

    x = qmodel.features[24][0].x
    B, C, W, H = x.shape
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
    x_bin = np.array(torch.round(x / 2**-7 + 0.5).view(-1).tolist()).astype(int)
    x_terms = booth.num_binary_terms(x, 2**-7)
    x_terms = x_terms.view(B*W*H, C//group_size, group_size).sum(2)
    x_terms = np.array(x_terms.cpu().view(-1).tolist()).astype(int)


    fig, axes = plt.subplots(2,2, figsize=(8, 4.5))

    min_b = abs(w_bin.min())
    bars = np.bincount(w_bin + min_b)
    bars = 100. * bars / bars.sum()
    axes[0,0].fill_between(np.arange(-min_b, len(bars)-min_b), bars, color='darkorange')
    axes[0,0].set_xlabel('Weight Values')
    axes[0,0].set_xticks([-128, 0, 128])
    axes[0,0].set_yticks([0, 2, 4, 6])
    axes[0,0].set_ylabel('Value\nFrequency')

    bars = np.bincount(x_bin)
    bars = 100. * bars / bars.sum()
    bars[0] = 1.9
    axes[0,1].fill_between(np.arange(len(bars)), bars, color='seagreen')
    axes[0,1].set_xlabel('Data Values')
    axes[0,1].set_xticks([0, 128])
    axes[0,1].set_yticks([0, 1, 1.8])
    axes[0,1].set_yticklabels([0, 1, 40])
    axes[0,1].set_ylim((0, 2))
    axes[0, 1].add_patch(matplotlib.patches.Rectangle((-20, 1.35), 330, 0.10,
                         color='white', clip_on=False, zorder=10000))
    axes[0, 1].add_patch(matplotlib.patches.Rectangle((-15.4, 1.45), 17, 0.02,
                         color='black', clip_on=False, ec=None, zorder=10001))
    axes[0, 1].add_patch(matplotlib.patches.Rectangle((-15.4, 1.34), 17, 0.02,
                         color='black', clip_on=False, ec=None, zorder=10001))

    axes[0, 1].add_patch(matplotlib.patches.Rectangle((288.5, 1.45), 17, 0.02,
                         color='black', clip_on=False, ec=None, zorder=10001))
    axes[0, 1].add_patch(matplotlib.patches.Rectangle((288.5, 1.34), 17, 0.02,
                         color='black', clip_on=False, ec=None, zorder=10001))

    bars = np.bincount(w_terms, minlength=8)
    bars = 100. * bars / bars.sum()
    axes[1,0].bar(np.arange(len(bars)), bars, color='darkorange', ec='k')
    axes[1,0].set_xlabel('# of Weight Terms')
    axes[1,0].set_xticks(np.arange(len(bars)))
    axes[1,0].set_yticks([0, 20, 40])
    axes[1,0].set_ylabel('# of Terms\nFrequency')
    axes[1,0].set_ylim((0, 45))

    bars = np.bincount(x_terms)
    bars = 100. * bars / bars.sum()
    axes[1,1].bar(np.arange(len(bars)), bars, color='seagreen', ec='k')
    axes[1,1].set_xlabel('# of Data Terms')
    axes[1,1].set_xticks(np.arange(len(bars)))
    axes[1,1].set_ylim((0, 45))

    plt.tight_layout()
    plt.savefig('figures/value-vs-terms.pdf', dpi=300, bbox_inches='tight')
    plt.clf()