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
plt = util.import_plt_settings(local_display=False)

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
                                                                num_train=800, num_val=500)

    def booth_test(model, weight_exp, data_exp, booth_values):
        model = copy.deepcopy(model)
        booth_weight_values = 2**weight_exp * torch.Tensor(booth_values).cuda(args.gpu)
        booth_data_values = 2**data_exp * torch.Tensor(booth_values).cuda(args.gpu)
        # booth_weight_values = torch.Tensor([2**i for i in range(-7, 1)] + [-2**i for i in range(-7, 1)] + [0]).cuda()
        model.cuda(0)

        # for pretrained resnet
        def replace_weights(model):
            for layer in model.children():
                if isinstance(layer, nn.Conv2d):
                    layer.weight.data = booth.booth_quant(layer.weight.data, booth_weight_values)
                    # layer.weight.data = net.quantize_cuda.forward(layer.weight.data, 2**-7, -127*2**-7, 127*2**-7, -127*2**-7)
                else:
                    replace_weights(layer)

        replace_weights(model)

        act_layer = nn.Sequential(
                        nn.ReLU(inplace=True),
                        booth.BoothQuant(booth_data_values),
                        # util.AverageTracker()
                    )

        for i in range(len(model.features)):
            if type(model.features[i]) == nn.ReLU:
                model.features[i] = act_layer

        criterion = nn.CrossEntropyLoss().cuda(0)
        tune_bn(train_loader, model, args)
        _, acc = util.validate(val_loader, model, criterion, args)
        del model

        return acc


    names = ['fixed-point', 'two', 'two-signed', 'three', 'three-signed']
    all_booth_values = [
        list(range(-127, 128)),
        booth.booth_two_values(0, 8, signed=False),
        booth.booth_two_values(0, 8, signed=True),
        booth.booth_three_values(0, 8, signed=False),
        booth.booth_three_values(0, 8, signed=True),
    ]
    weight_exps = [-7, -6]
    data_exps = [-5, -4, -3]
    accs = []
    for booth_values in all_booth_values:
        acc_type = []
        for weight_exp in weight_exps:
            for data_exp in data_exps:
                acc = booth_test(model, weight_exp, data_exp, booth_values)
                acc_type.append(acc)
    
    for name, acc_type in zip(names, accs):
        print('{}: {}'.format(name, acc_type))