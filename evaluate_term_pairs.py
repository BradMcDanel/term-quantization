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

def get_term_count(model, minlength=None):
    w = model.features[12].weight.data
    x = model.features[11][1].x
    B, C, W, H = x.shape
    F = w.shape[0]
    x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)
    x_terms = booth.num_hese_terms(x, 2**-8)
    x_terms = x_terms.view(B*W*H, C)
    w = w.view(F, C)
    w_terms = booth.num_hese_terms(w, w_sfs[4])
    w_terms = w_terms.view(F, C)

    all_res = []
    for i in range(0, C, group_size):
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
                                                                num_train=128, num_val=64)

    group_size = 8
    stat_terms = [4, 3]
    terms = [group_size*4, 12]

    bcs = []
    for i, (term, stat_term) in enumerate(zip(terms, stat_terms)):
        qmodel = copy.deepcopy(model).cpu()
        w_sfs = util.quantize_layers(qmodel, bits=8)
        qmodel.cuda()
        qmodel = models.convert_model(qmodel, w_sfs, stat_term, 1, term, group_size, stat_term,
                                      1, term, group_size, models.data_stationary_point(model))

        criterion = nn.CrossEntropyLoss().cuda()
        util.add_average_trackers(qmodel)
        qmodel.cuda()
        _, acc = util.validate(val_loader, qmodel, criterion, args, verbose=True)
        if i == 0:
            bc = get_term_count(qmodel, minlength=group_size*stat_term*stat_term)[1:]
        else:
            bc = get_term_count(qmodel)[1:]
        bcs.append(bc)

    fill_colors = ['cornflowerblue', 'tomato']
    names = ['Original', r'Term Revealing ($\alpha=1.5$)']
    fig, axes = plt.subplots(2)
    for i, (ax, bc, fill_color, name) in \
        enumerate(zip(axes, bcs, fill_colors, names)):
        xs = np.arange(1, len(bc) + 1)
        bc = 100. * (bc / bc.sum())
        ax.plot(xs, bc, '-', color='k', linewidth=1.0)
        ax.fill_between(xs, bc, color=fill_color, edgecolor=fill_color, label=name)
        ax.set_xlim(0, len(bcs[0])+1)
        ax.legend(loc=0)

        if i == 0:
            ax.annotate('Largest Observed (72)', 
                xy=(72, -0.1),  
                textcoords='data', xytext=(44, 1.6),
                clip_on=True,
                arrowprops=dict(arrowstyle="->"))

            ax.annotate('Max Possible (128)', 
                xy=(127.9, -0.1),  
                textcoords='data', xytext=(80, 3),
                clip_on=True,
                arrowprops=dict(arrowstyle="->"))
        else:
            ax.annotate('Max Possible (36)', 
                xy=(35.5, -0.1),  
                textcoords='data', xytext=(32, 2.5),
                clip_on=True,
                arrowprops=dict(arrowstyle="->"))

    axes[1].set_xlabel('Number of Term Pairs')
    # axes[0].set_ylabel('Frequency (%)')
    # axes[1].set_ylabel('Frequency (%)')
    fig.text(0.04, 0.5, 'Frequency (%)', rotation=90, ha='center', va='center', fontsize=18)

    # plt.legend(loc=0)
    # plt.yscale('log')
    axes[0].set_title('Term Pair Frequency per Group of 8')
    plt.savefig('figures/shiftnet.png', dpi=300, bbox_inches='tight')
    plt.clf()
