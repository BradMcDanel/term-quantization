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
import matplotlib.ticker as mticker


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
    stat_term = 4
    term = 16

    criterion = nn.CrossEntropyLoss().cuda()

    omodel = copy.deepcopy(model).cpu()
    _ = util.quantize_layers(omodel, bits=8)
    util.add_uniform_quant(omodel, 8, 2**-7)
    util.add_average_trackers(omodel, before_layer=nn.Conv2d)
    omodel.cuda(args.gpu)
    _ = util.validate(val_loader, omodel, criterion, args, verbose=False)

    # quantized model
    qmodel = copy.deepcopy(model).cpu()
    _ = util.quantize_layers(qmodel, bits=8)
    util.add_uniform_quant(qmodel, 7, 2**-7)
    util.add_average_trackers(qmodel, before_layer=nn.Conv2d)
    qmodel.cuda(args.gpu)
    _ = util.validate(val_loader, qmodel, criterion, args, verbose=False)

    # term revealing model
    trmodel = copy.deepcopy(model).cpu()
    w_sfs = util.quantize_layers(trmodel, bits=8)
    util.add_uniform_quant(trmodel, 8, 2**-7)
    trmodel.cuda(args.gpu)
    trmodel = models.convert_model(trmodel, w_sfs, stat_term, 1, term, group_size, stat_term,
                                   1, term, group_size, 1)
    util.add_average_trackers(trmodel, before_layer=nn.Conv2d)
    _ = util.validate(val_loader, trmodel, criterion, args, verbose=False)


    o_layers = util.get_average_trackers(omodel)
    q_layers = util.get_average_trackers(qmodel)
    tr_layers = util.get_average_trackers(trmodel)

    qerrors = []
    trerrors = []
    for olayer, qlayer, trlayer in zip(o_layers, q_layers, tr_layers):
        error = (olayer.x - qlayer.x).abs().mean()
        qerrors.append(error)
    
        error = (olayer.x - trlayer.x).abs().mean()
        trerrors.append(error)
    
    plt.figure(figsize=(5, 4.5))
    bar_width = 0.4
    names = ['7-bit Quantization', r'TR ($g=8$, $\alpha=2$)']
    colors = ['darkseagreen', 'tomato']
    markers = ['o', '^']
    x_ticks = np.arange(1, 18)
    bars = [qerrors, trerrors]
    for i in range(len(bars)):
        plt.plot(x_ticks, 100*np.array(bars[i][1:]), '-o', linewidth=2.5, ms=6,
                 marker=markers[i], markeredgecolor='k', color=colors[i],
                 label=names[i])

    plt.title('(a) Quantization Error')
    plt.xlabel('Convolution Layer Activations')
    plt.ylabel('Avg. Quantization Error')
    plt.text(-2.5, 2.8, '1e-2', fontsize=16)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    plt.legend(loc=0)
    plt.savefig('figures/comp-quant-error.pdf', dpi=300, bbox_inches='tight')
    plt.clf()