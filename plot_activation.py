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


import util
plt = util.import_plt_settings(local_display=False)
import models

def imshow_save(filename, data, vmax):
    plt.imshow(data, vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    plt.axis('off')
    cbar = plt.colorbar(orientation='horizontal')
    cbar.set_ticks(list(range(-vmax, vmax+1, 64)))
    plt.savefig(filename, bbox_inches='tight', dpi=600)
    plt.clf()


def evaluate(loader, model, args, name):
    # switch to evaluate mode
    model.eval()
    average_trackers = util.get_average_trackers(model)
    conflict_mats = []
    data_shapes = []
    num_samples = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            num_samples += len(target)
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)
            _ = model(images)
            for k, tracker in enumerate(average_trackers):
                data = tracker.x.clone()
                data[data != 0] = 1

                # only care about conv layers for now
                if len(data.shape) < 4:
                    continue

                _, C, W, H = data.shape
                data = data.permute(1, 0, 2, 3).contiguous().view(C, -1)
                # first batch -- generate conflict mat
                if i == 0:
                    data_shapes.append((C, W, H))
                    conflict_mat = torch.mm(data, torch.transpose(data, 0, 1))
                    conflict_mats.append(conflict_mat)
                else:
                    conflict_mats[k] += torch.mm(data, torch.transpose(data, 0, 1))

    for i, conflict_mat in enumerate(conflict_mats):
        # remove diag 
        C, W, H = data_shapes[i]
        conflict_mat[range(len(conflict_mat)), range(len(conflict_mat))] = 0
        conflict_mat /= num_samples*W*H
    
    return conflict_mats, average_trackers


def rle(inarray):
    """ run length encoding. Partial credit to R rle function. 
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

def get_conflict_score(conflict_mat, columns):
    from itertools import combinations 
    conflict_score = 0.0
    for i, j in combinations(columns, 2):
        conflict_score += conflict_mat[i][j].item()
    return conflict_score

def first_fit(conflict_mat, size_idxs, max_conflict_score=0.01, max_columns=8):
    bins = []
    for column in size_idxs:
    # for column in range(len(conflict_mat)):
        for bin_elems in bins:
            # cannot add to column as reached max multiplexing size
            if len(bin_elems) == max_columns:
                continue
            potential_bin = bin_elems + [column]
            conflict_score = get_conflict_score(conflict_mat, potential_bin)
            # valid bin found, add column to bin then break
            if conflict_score < max_conflict_score:
                bin_elems.append(column)
                break
        # no valid bin found, so create new bin
        else:
            bins.append([column])
    return bins

def tune_bn(loader, model, args):
    # switch to evaluate mode
    model.train()
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = images.cuda(0, non_blocking=True)
            _ = model(images)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('resume', help='path to load model')
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

    # optionally resume from a checkpoint
    # if os.path.isfile(args.resume):
    #     print("=> loading checkpoint '{}'".format(args.resume))
    #     checkpoint = torch.load(args.resume)
    #     tmp_state_dict = dict(checkpoint['state_dict'])
    #     state_dict = {}
    #     for key in tmp_state_dict.keys():
    #         if 'module.' in key:
    #             new_key = key.replace('module.', '')
    #         else:
    #             new_key = key
    #         state_dict[new_key] = tmp_state_dict[key]
    #     state_dict = OrderedDict(state_dict)
    #     model.load_state_dict(state_dict)
    #     print("=> loaded checkpoint '{}' (epoch {})"
    #           .format(args.resume, checkpoint['epoch']))
    # else:
    #     print("=> no checkpoint found at '{}'".format(args.resume))
    #     assert False

    cudnn.benchmark = True

    train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
                                                                num_train=80, num_val=64)
    
    def get_layer_sizes(model):
        model = copy.deepcopy(model)
        util.add_average_trackers(model)
        model.cuda(0)
        _, trackers = evaluate(train_loader, model, args, 'train')
        conv_layers = util.get_layers(model, [nn.Conv2d])
        weight_sizes, data_sizes = [], []
        for i in range(len(conv_layers)):
            B, C, W, H = conv_layers[i].weight.shape
            data_w = trackers[i].x.shape[1]
            data_h = trackers[i].x.shape[2]*trackers[i].x.shape[3]
            weight_w = C*W*H
            weight_h = B
            weight_sizes.append((weight_w, weight_h))
            data_sizes.append((data_w, data_h))
        
        return weight_sizes, data_sizes

    util.add_average_trackers(model)
    model.cuda(0)
    train_conflicts, trackers = evaluate(val_loader, model, args, 'train')

    for i in range(20):
        # data = trackers[0].x[0, i].view(112, 112)
        data = trackers[4].x[0, i].view(56, 56)
        data /= 2**-8
        data = data.tolist()
        imshow_save('figures/imgs/fm-{}.png'.format(i), data, 128)

    weight = model.features[0].weight.view(64, -1)
    weight /= 2**-8
    weight = weight.tolist()
    imshow_save('figures/imgs/weight-mat.png', weight, 128)