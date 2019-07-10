import argparse
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
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
SMALL_SIZE = 13
MEDIUM_SIZE = 17
BIGGER_SIZE = 20
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

import models
import cgm
import numpy as np

def validate(val_loader, model, args):
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)
            output = model(images)

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

class AverageTracker(nn.Module):
    def __init__(self):
        super(AverageTracker, self).__init__()
        self.count = 0
        self.register_buffer('data', None)
        self.register_buffer('x', None)

    def forward(self, x):
        if self.data is None:
            self.data = torch.zeros(x.size()[1:]).cuda(0)
            self.zeros = torch.zeros(x.size()[1:]).cuda(0)
            self.x = torch.zeros(x.size()).cuda(0)
        self.x = x
        self.data += x.sum(0)
        self.count += x.size(0)
        self.zeros += (x == 0).float().sum(0)
        return x
    
    def avg_data(self):
        return self.data / self.count
    
    def channel_zeros(self):
        C, W, H = self.zeros.shape
        return self.zeros.sum((1, 2)) / float(W*H*self.count)

def get_conflict_score(conflict_mat, columns):
    from itertools import combinations 
    conflict_score = 0.0
    for i, j in combinations(columns, 2):
        conflict_score += conflict_mat[i][j].item()
    return conflict_score

def first_fit(conflict_mat, max_conflict_score=0.01, max_columns=8):
    bins = []
    for column in range(len(conflict_mat)):
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
        num_train = 1024 # just a small number for this test
        num_val = 1024 # just a small number for this test
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset = InMemoryImageNet(train_path, num_val,
                                transforms=transforms.Compose([
                                    msgpack_load,
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize,
                                ]))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=False, drop_last=False,
                                                   num_workers=args.workers)
        train_loader.num_samples = num_train
 

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

    layers = []
    for l in model.features:
        if type(l) == cgm.CGM or type(l) == nn.ReLU:
            layers.extend([l, AverageTracker()])
        else:
            layers.append(l)

    model.features = nn.Sequential(*layers)
    model.cuda(0)
    validate(train_loader, model, args)

    assert False
    validate(val_loader, model, args)

    assert False

    layer_bin_data = []
    conflict_scores = np.linspace(0.0, 1.0, 50)
    for k, (lidx, size) in enumerate([(3, 64), (8, 192), (13, 384), (17, 256), (21, 256)]):
        img = model.features[lidx].x[0]
        img = img[:64]
        C, W, H = img.shape
        save_image(make_grid(img.view(C, 1, W, H)), 'figures/img-{}-{}.png'.format(args.arch, k+1))
        # invert so 0's == 1 and everything else == 0
        zero_runs = model.features[lidx].x.permute(1,0,2,3).contiguous().view(size, -1)
        zero_runs[zero_runs != 0] = 2
        zero_runs[zero_runs == 0] = 1
        zero_runs[zero_runs == 2] = 0

        data = []
        for i in range(20):
            run_length, _, vals = rle(zero_runs[i].cpu().numpy())
            run_length = run_length[vals==1]  # only interested in zero runs for now
            if np.mean(run_length) < 200:
                data.append(run_length.tolist())

        plt.boxplot(data, showfliers=False)
        plt.title('AlexNet Layer {}'.format(k+1))
        plt.xlabel('Channel Index')
        plt.ylabel('Zero Run Length (boxplot)')
        plt.tight_layout()
        plt.savefig('figures/alexnet-box-{}.png'.format(k+1), dpi=300)
        plt.clf()

        data = model.features[lidx].x.clone()
        data[data != 0] = 1
        B, C, W, H = data.shape
        data = data.permute(1, 0, 2, 3).contiguous().view(C, -1)
        sim_mat = torch.mm(data, torch.transpose(data, 0, 1))
        # remove diag 
        sim_mat[range(len(sim_mat)), range(len(sim_mat))] = 0
        sim_mat /= B*W*H

        plt.imshow(sim_mat.tolist())
        plt.colorbar()
        plt.title('AlexNet Layer {}'.format(k+1))
        plt.tight_layout()
        plt.savefig('figures/sim-mat-{}'.format(k+1), dpi=300)
        plt.clf()
        
        total_bins = []
        for conflict_score in conflict_scores:
            print('Processing.. {}'.format(conflict_score))
            bins = first_fit(sim_mat, conflict_score)
            total_bins.append(len(bins))
        layer_bin_data.append(total_bins)

    for lidx, total_bins in enumerate(layer_bin_data):
        plt.plot(conflict_scores, total_bins, label='Layer {}'.format(lidx))

    plt.xlabel('Max Conflict Score')
    plt.ylabel('Number of Data Channels')
    plt.legend(loc=0)
    plt.xlim((-0.05, 1.05))
    plt.tight_layout()
    plt.savefig('figures/first_fit_conflicts.png', dpi=300)


    # zeros = []
    # for i, l in enumerate(model.features):
    #     if type(l) == AverageTracker:
    #         z = l.channel_zeros()
    #         zeros.append(z.tolist())
    #         #plt.plot(torch.sort(z)[0].tolist(), '-o', linewidth=2)
    #         #plt.savefig('{}-{}-zeros.png'.format(args.arch, i))
    #         #plt.clf()

    # with open('{}.pkl'.format(args.arch), 'wb') as f:
    #     pickle.dump(zeros, f)
