import argparse
import os
import random
import shutil
import time
import warnings

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

import models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')


args = parser.parse_args()
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
if not args.evaluate:
    train_dataset = InMemoryImageNet(train_path, num_train,
                            transforms=transforms.Compose([
                                msgpack_load,
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                shuffle=True, drop_last=False,
                                                num_workers=args.workers)
    train_loader.num_samples = num_train
else:
    # do not bother loading train_dataset into memory if we are just
    # going to evaluate the model on the validation set (saves times)
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
