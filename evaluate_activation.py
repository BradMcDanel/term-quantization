import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict

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

import models
import cgm

def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)
            output = model(images)

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
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


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
        state_dict = dict(checkpoint['state_dict'])
        for key in state_dict.keys():
            if '.module' in key:
                new_key = key.replace('.module', '')
                state_dict[new_key] = state_dict.pop(key)
        state_dict = OrderedDict(state_dict)
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        assert False

    cudnn.benchmark = True

    ### begin custom data loader
    # using custom msgpack data loader due to slow disk, replace with standard
    # pytorch ImageFolder loader for equivalent results (shown below in else)
    if args.msgpack_loader:
        import pickle
        import msgpack
        from PIL import Image
        import io

        def msgpack_load(x):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = Image.open(io.BytesIO(x)).convert('RGB')
            return x

        class InMemoryImageNet(Dataset):
            def __init__(self, path, num_samples, transforms):
                self.path = path
                self.num_samples = num_samples
                self.transforms = transforms
                self.samples = []
                f = open(self.path, "rb")
                for i, sample in enumerate(msgpack.Unpacker(f, use_list=False, raw=True)):
                    x, label = sample
                    x = pickle.loads(x)
                    self.samples.append((x, label))
                    if i == self.num_samples - 1:
                        break
                f.close()
                
            def __getitem__(self, index):
                x, y = self.samples[index]
                x = self.transforms(x)
                return (x, y)

            def __len__(self):
                return self.num_samples

        train_path = os.path.join(args.data, 'imagenet-msgpack', 'ILSVRC-train.bin')
        val_path = os.path.join(args.data, 'imagenet-msgpack', 'ILSVRC-val.bin')
        num_train = 1281167
        num_val = 50000
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset, train_loader = None, None
        val_dataset = InMemoryImageNet(val_path, num_val,
                                transforms=transforms.Compose([
                                    msgpack_load,
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                                 shuffle=False, drop_last=False,
                                                 num_workers=args.workers)
        val_loader.num_samples = num_val
    ### end custom data loader

    # Use standard PyTorch Dataloader
    else:  
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)

    class AverageTracker(nn.Module):
        def __init__(self):
            super(AverageTracker, self).__init__()
            self.count = 0
            self.register_buffer('data', None)
            self.register_buffer('x', None)

        def forward(self, x):
            if self.data is None:
                self.data = torch.zeros(x.size()[1:]).cuda(0)
                self.x = torch.zeros(x.size()).cuda(0)
            self.x = x
            self.data += x.sum(0)
            self.count += x.size(0)
            return x
        
        def avg_data(self):
            return self.data / self.count


    layers = []
    for l in model.features:
        if type(l) == cgm.CGM or type(l) == nn.ReLU:
            layers.extend([l, AverageTracker()])
        else:
            layers.append(l)

    model.features = nn.Sequential(*layers)
    model.cuda(0)
    validate(val_loader, model, args)

    save_image(make_grid(model.features[3].x[0].view(64, 1, 55, 55)), '{}-1.png'.format(args.arch))

