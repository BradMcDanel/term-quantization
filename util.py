import warnings
import pickle
import io
import time
import os
import shutil
import math

import msgpack
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from q_utils import *
from shift import Shift
import booth

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    beta = bn.weight
    gamma = bn.bias
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)
    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                            conv.out_channels,
                            conv.kernel_size,
                            conv.stride,
                            conv.padding,
                            bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def import_plt_settings(local_display=True):
    import matplotlib
    if not local_display:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    SMALL_SIZE = 14
    TICK_SIZE = 15
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=TICK_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=TICK_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    return plt

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def get_layers(model, layer_types):
    layers = []
    for child in model.children():
        if any([isinstance(child, l) for l in layer_types]):
            layers.append(child)
        else:
            child_layers = get_layers(child, layer_types)
            layers.extend(child_layers)
    
    return layers

def cycle_time(stationary_mat, piped_mat, sa_size):
    '''
    Computes systolic array cycle time (does not include possible I/O cost).
    Bit-parallel computation assumed.
    '''
    sw, sh = stationary_mat
    pw, ph = piped_mat

    assert pw == sw, 'stationary and piped width must match. Got {} and {}.'.format(sw, pw)

    cycles = 0
    num_tiles = math.ceil(sw / sa_size) * math.ceil(sh / sa_size)
    for _ in range(num_tiles):
        # weight loading (sa_size) + data loading (ph) + skew (sa_size*2)
        cycles += sa_size + ph + sa_size*2
    
    return cycles

def get_imagenet(args, train_name='ILSVRC-train.bin', num_train=1281167, num_val=50000):
    ### begin custom data loader
    # using custom msgpack data loader due to slow disk, replace with standard
    # pytorch ImageFolder loader for equivalent results (shown below in else)
    if args.msgpack_loader:
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

        train_path = os.path.join(args.data, 'imagenet-msgpack', train_name)
        val_path = os.path.join(args.data, 'imagenet-msgpack', 'ILSVRC-val.bin')
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
        train_sampler = None
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

    return train_loader, train_sampler, val_loader

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


def validate(val_loader, model, criterion, args, verbose=True, pct=1.0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    eval_samples = round(pct * val_loader.dataset.num_samples)
    curr_samples = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            curr_samples += len(target)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and verbose:
                progress.display(i)
            
            if curr_samples >= eval_samples:
                break


        # TODO: this should also be done with the ProgressMeter
        if verbose:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename + '.best')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

def add_average_trackers(model, before_layer):
    for child_name, child in model.named_children():
        if isinstance(child, before_layer):
            setattr(model, child_name, nn.Sequential(AverageTracker(), child))
        else:
            add_average_trackers(child, before_layer)

def get_average_trackers(model):
    average_trackers = []
    for child in model.children():
        if isinstance(child, AverageTracker):
            average_trackers.append(child)
        else:
            trackers = get_average_trackers(child)
            average_trackers.extend(trackers)
    
    return average_trackers


def quantize_layer(layer, bits):
    w = layer.weight.data.view(-1).numpy()
    alpha = find_clip_mmse(w, bits)
    sf = 1.0 / symmetric_linear_quantization_scale_factor(bits, alpha)
    w = distiller_quantize(w, bits, alpha)
    w = torch.from_numpy(w).view_as(layer.weight.data)
    layer.weight.data = w   
    return sf

def quantize_layers(model, bits):
    sfs = []
    for layer in get_layers(model, [nn.Conv2d]):
        w = layer.weight.data.view(-1).numpy()
        alpha = find_clip_mmse(w, bits)
        sf = 1.0 / symmetric_linear_quantization_scale_factor(bits, alpha)
        w = layer.weight.data
        w = sf * torch.floor((w / sf) + 0.5)
        layer.weight.data = w
        sfs.append(sf)
    return sfs


class UniformQuant(nn.Module):
    def __init__(self, bits, sf):
        super(UniformQuant, self).__init__()
        self.bits = bits
        self.sf = sf
        self.minv = -sf * 2**bits
        self.maxv = sf * 2**bits
    
    def forward(self, x):
        x = self.sf * torch.round(x / self.sf + 0.5)
        x = torch.clamp(x, self.minv, self.maxv)
        return x

def add_uniform_quant(model, bits, sf=2**-7):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Sequential(child, UniformQuant(bits, sf)))
        else:
            add_uniform_quant(child, bits, sf)

    
def distiller_quantize(x, num_bits, alpha):
    min_q_val, max_q_val = get_quantized_range(num_bits, signed=True)
    scale = symmetric_linear_quantization_scale_factor(num_bits, alpha)
    q = linear_quantize_clamp(torch.from_numpy(x), scale, min_q_val, max_q_val)
    x = linear_dequantize(q, scale)
    return x.numpy()

def mse_histogram_clip(bin_x, bin_y, num_bits, alpha):
   # Clipping error: sum over bins outside the clip threshold
   idx = np.abs(bin_x) > alpha
   mse = np.sum((np.abs(bin_x[idx]) - alpha)**2 * bin_y[idx])
   # Quantization error: sum over bins inside the clip threshold
   idx = np.abs(bin_x) <= alpha
   bin_xq = distiller_quantize(bin_x[idx], num_bits, alpha)
   mse += np.sum((bin_x[idx] - bin_xq)**2 * bin_y[idx])
   return mse

def find_clip_mmse(values, num_bits):
    # Build histogram
    max_abs = np.max(np.abs(values))
    bin_y, bin_edges = np.histogram(values, bins=201, density=True)
    bin_x = 0.5*(bin_edges[:-1] + bin_edges[1:])
    bin_y /= np.sum(bin_y)

    alphas = np.arange(0.01, 1, 0.01) * max_abs
    mses = [ mse_histogram_clip(bin_x, bin_y, num_bits, alpha)
             for alpha in alphas ]

    alpha_best = alphas[np.argmin(mses)]
    return alpha_best

def relu_to_booth(model, sf, groups, terms):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Sequential(
                nn.ReLU(inplace=True),
                booth.Radix2ModGroup(sf, groups, terms)
            ))
        elif isinstance(child, nn.ReLU6):
            setattr(model, child_name, nn.Sequential(
                nn.ReLU6(inplace=True),
                booth.Radix2ModGroup(sf, groups, terms)
            ))
        else:
            relu_to_booth(child, sf, groups, terms)
        