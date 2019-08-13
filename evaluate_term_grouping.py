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


import booth
import models
import util
plt = util.import_plt_settings(local_display=False)

def tune_bn(loader, model, args):
    # switch to train mode (update bn mean/std)
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.cuda(args.gpu, non_blocking=True)
            _ = model(images)

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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--msgpack-loader', dest='msgpack_loader', action='store_true',
                    help='use custom msgpack dataloader')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


if __name__=='__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    cudnn.benchmark = True

    train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
                                                                num_train=8000, num_val=5000)
    def replace_weights(model, group_size, num_keep_terms):
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                if layer.kernel_size[0] != 1:
                    continue

                weight = booth.booth_cuda.radix_2_mod(layer.weight.data, 2**-15,
                                                      group_size, num_keep_terms)
                layer.weight.data = weight
            else:
                replace_weights(layer, group_size, num_keep_terms)

    def booth_test(model, num_weight_exps, num_data_exps, group_size, mode='weight'):
        model = copy.deepcopy(model)
        model.cuda(0)

        if mode == 'data':
            replace_weights(model, 1, num_weight_exps)
        else:
            replace_weights(model, group_size, group_size*num_weight_exps)


        if mode == 'data':
            act_layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.Radix2ModGroup(2**-15, group_size, group_size*num_data_exps),
                        )
        else:
            act_layer = nn.Sequential(
                            nn.ReLU(inplace=True),
                            booth.Radix2ModGroup(2**-15, 1, group_size*num_data_exps),
                        )

        for i in range(len(model.features)):
            if type(model.features[i]) == nn.ReLU:
                model.features[i] = act_layer

        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda(0)
        # tune_bn(train_loader, model, args)
        _, acc = util.validate(val_loader, model, criterion, args, verbose=False)
        return model, acc
     
    # TODO: fold in bn and retest with 3 terms per weight

    values = list(range(-512, 512))
    num_columns = 2
    num_cycles = 2
    num_data_exps = [8]
    num_weight_exps = [3]
    group_sizes = [1, 4, 8]
    accs = []
    mode = 'data'
    for group_size in group_sizes:
        acc_row = [] 
        for num_weight_exp, num_data_exp in product(num_weight_exps, num_data_exps):
            if mode == 'weight':
                we = num_weight_exp*group_size
                de = num_data_exp
            else:
                we = num_weight_exp
                de = num_data_exp
            qmodel, acc = booth_test(model, we, de, group_size, mode)
            print(group_size, we, de, acc.item())
            acc_row.append(acc.item())
        accs.append(acc_row)
    
    colors = ['r', 'g', 'b']
    for i, (acc_row, group_size) in enumerate(zip(accs, group_sizes)):
        if i == 0:
            plt.plot(num_data_exps, acc_row, '-o', linewidth=2.5, color=colors[i], label='No Grouping')
            continue
        plt.plot(num_data_exps, acc_row, '--o', linewidth=2.5, color=colors[i],
                 label='Group({})-values'.format(group_size))


    plt.legend(loc=0)
    plt.xticks([2, 3, 4])
    plt.xlabel('Average Number of Data Terms')
    plt.ylabel('Classification Accuracy (%)')
    plt.savefig('figures/pow2-data-grouping.png', dpi=300)
    plt.clf()