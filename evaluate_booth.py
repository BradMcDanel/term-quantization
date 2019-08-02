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


import booth
import cgm
import models
import util
plt = util.import_plt_settings(local_display=True)

def tune_bn(loader, model, args):
    # switch to train mode (update bn mean/gamma)
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
    model = models.__dict__[args.arch]()

    # optionally resume from a checkpoint
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        tmp_state_dict = dict(checkpoint['state_dict'])
        state_dict = {}
        for key in tmp_state_dict.keys():
            if 'module.' in key:
                new_key = key.replace('module.', '')
            else:
                new_key = key
            state_dict[new_key] = tmp_state_dict[key]
        state_dict = OrderedDict(state_dict)
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        assert False

    cudnn.benchmark = True

    train_loader, train_sampler, val_loader = util.get_imagenet(args, 'ILSVRC-train-chunk.bin',
                                                                num_train=8000, num_val=50000)
    # for pretrained resnet
    def replace_weights(model, weight_exp, terms, group_size, num_keep_terms):
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                C = layer.weight.shape[1]
                print(C, group_size)
                single = booth.weight_single_quant(layer.weight.data, 2**weight_exp,
                                                   terms, num_keep_terms // group_size)
                group = booth.weight_group_quant(layer.weight.data, 2**weight_exp,
                                                 terms, min(C, group_size), num_keep_terms)
                # print((single.view(-1) - layer.weight.data.view(-1)).abs().sum(),
                #       (group.view(-1) - layer.weight.data.view(-1)).abs().sum())
                if layer.kernel_size[0] != 1:
                    continue
                else:
                    layer.weight.data = group
            else:
                replace_weights(layer, weight_exp, terms, group_size, num_keep_terms)

    def booth_test(model, weight_exp, data_exp, num_weight_exps, num_data_exps, group_size, terms):
        model = copy.deepcopy(model)
        model.cuda(0)

        replace_weights(model, weight_exp, terms, group_size, num_weight_exps)

        act_layer = nn.Sequential(
                        nn.ReLU(inplace=True),
                        booth.BoothQuant(2**data_exp, num_data_exps),
                        # util.AverageTracker()
                    )

        for i in range(len(model.features)):
            if type(model.features[i]) == nn.ReLU:
                model.features[i] = act_layer

        model.cuda()
        criterion = nn.CrossEntropyLoss().cuda(0)
        tune_bn(train_loader, model, args)
        _, acc = util.validate(val_loader, model, criterion, args, verbose=False)
        return model, acc
     
    values, value_powers = booth.two_powers(1, 9)
    values = list(range(-512, 512))
    num_columns = 2
    num_cycles = 2
    num_data_exp = 4
    # num_weight_exps = [1, 2, 3, 4]
    # group_sizes = [1, 2, 4, 8, 16, 32]
    num_weight_exps = [1]
    group_sizes = [16, 32]
    weight_terms = booth.min_power_rep(8, 6)
    weight_terms = booth.pad_torch_mat(weight_terms, min_dim=6).int().cuda()
    accs = []
    names = ['G=1', 'G=2', 'G=4', 'G=8', 'G=16', 'G=32']
    for group_size in group_sizes:
        acc_row = [] 
        for num_weight_exp in num_weight_exps:
            we = num_weight_exp*group_size
            qmodel, acc = booth_test(model, -8, -7, we, num_data_exp, group_size, weight_terms)
            print(group_size, we, acc)
            acc_row.append(acc.item())
        accs.append(acc_row)
    
    for name, acc_row, group_size in zip(names, accs, group_sizes):
        plt.plot(num_weight_exps, acc_row, linewidth=2.5, label=name)
    plt.legend(loc=0)
    plt.xlabel('Number of Weight Terms')
    plt.ylabel('Classification Accuracy (%)')
    plt.yscale('log')
    plt.savefig('figures/pow2-grouping.png', dpi=300)
    plt.clf()
    assert False

    # model.cuda(args.gpu)
    # weight_exp = -8
    # min_exp, max_exp = 1, 9
    # # weight_exp = -7
    # # min_exp, max_exp = 0, 8
    # w = model.features[4].weight.data
    # float_vals = w.view(-1).tolist()
    # values, value_powers = booth.two_powers(min_exp, max_exp)
    # booth_weight_values = 2**weight_exp * torch.Tensor(values).cuda(args.gpu)
    # replace_weights(model, booth_weight_values)
    # vals = (model.features[4].weight.data.view(-1) / 2**weight_exp).long().tolist()
    # exp_bins = [0] * ((max_exp - min_exp) * 2 + 1)

    # power_rep = []
    # num_exps = [0] * 8
    # for v in vals:
    #     powers = booth.encode(v)
    #     print(v, powers)
    #     powers = booth.get_powers(v, value_powers, max_exp)
    #     for power in powers:
    #         exp_bins[power + max_exp - 1] += 1

    #     if powers[0] == 0:
    #         power_rep.append(0)
    #         num_exps[0] += 1
    #     else:
    #         power_rep.append(len(powers))
    #         num_exps[len(powers)] += 1
    
    # power_rep = torch.Tensor(power_rep).view_as(w)
    
    # plt.bar([0, 1, 2], num_exps, width=0.8, edgecolor='black')
    # plt.xlabel('Number of Powers-of-Two Required per Weight')
    # plt.ylabel('Frequency')
    # plt.xticks([0, 1, 2])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.savefig('figures/freq/num_exp_freq.png', dpi=300)
    # plt.clf()

    # plt.bar(np.arange(len(exp_bins)), exp_bins, width=0.8, edgecolor='black')
    # plt.xlabel('Power-of-Two Weight Exponenet')
    # plt.ylabel('Frequency')
    # plt.xticks(np.arange(len(exp_bins)), [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.savefig('figures/freq/exp_freq.png', dpi=300)
    # plt.clf()

    # plt.hist(float_vals, bins=100, edgecolor='black')
    # plt.xlabel('Float Weight Value')
    # plt.ylabel('Frequency')
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.savefig('figures/freq/float_freq.png', dpi=300)
    # plt.clf()

    # plt.hist([v * 2**weight_exp for v in vals], bins=100, edgecolor='black')
    # plt.xlabel('Weight Value (after quantization)')
    # plt.ylabel('Frequency')
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.savefig('figures/freq/fixed_freq.png', dpi=300)
    # plt.clf()