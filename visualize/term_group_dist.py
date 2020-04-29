import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import cnn_models
import tr_layer

import util
import visualize
plt = visualize.import_plt_settings(local_display=True)
import matplotlib.patches as patches

import bit_utils


class Tracker(nn.Module):
    def __init__(self):
        super(Tracker, self).__init__()
        self.x = None

    def forward(self, x):
        self.x = x
        return x

def add_trackers(model):
    curr_layer = 0
    for name, layer in model.named_modules():
        if isinstance(layer, tr_layer.TRConv2dLayer):
            module_keys = name.split('.')
            module = model
            for k in module_keys[:-1]:
                module = module._modules[k]

            layer = nn.Sequential(
                Tracker(),
                layer
            )

            module._modules[module_keys[-1]] = layer
            curr_layer += 1

    return model

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Group Distribution Example')
    parser.add_argument('val_dir', help='path to validation data folder')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                        choices=cnn_models.model_names(),
                        help='model architecture: ' +
                        ' | '.join(cnn_models.model_names()) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose flag')

    args = parser.parse_args()
    val_loader = util.get_imagenet_validation(args)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    model = cnn_models.__dict__[args.arch](pretrained=True).cuda(args.gpu)
    weight_bits = 9
    group_size = 8
    data_bits = 9
    data_terms = 9
    weight_terms = 9
    group_size = 1

    # replace Conv2d with TRConv2dLayer
    tr_params = cnn_models.static_conv_layer_settings(model, weight_bits, group_size, weight_terms)
    qmodel = cnn_models.convert_model(model, tr_params, data_bits, data_terms)
    qmodel.cuda(args.gpu)
    util.validate(val_loader, qmodel, criterion, args, verbose=args.verbose, pct=0.05)
    tr_layer.set_tr_tracking(qmodel, False)

    qmodel = add_trackers(qmodel)
    util.validate(val_loader, qmodel, criterion, args, pct=0.001)

    layer = qmodel.layer2[0].downsample[0]

    # get x_bits
    x = layer[0].x[:2]
    xq = layer[1].input_quant(x)
    xq_bits = bit_utils.expand_binary_bits(xq, layer[1].input_quant.sf)
    B, C, W, H, xB = xq_bits.shape
    xq_bits = xq_bits.permute(0, 4, 1, 2, 3).contiguous()
    xq_bits = xq_bits.view(-1, C, W, H)

    # get w_bits
    wq_bits = bit_utils.expand_binary_bits(layer[1].conv.weight, layer[1].w_sf)
    N, C, kW, kH, wB = wq_bits.shape
    wq_bits = wq_bits.permute(0, 4, 1, 2, 3).contiguous()
    wq_bits = wq_bits.view(-1, C, kW, kH)

    # group size settings
    group_size = 16
    r_sum = torch.zeros(B, N, W*H)
    r_bits = F.conv2d(xq_bits[:, :group_size], wq_bits[:, :group_size],
                      stride=layer[1].conv.stride,
                      padding=layer[1].conv.padding)
    r_bits = r_bits.view(B, xB, N, wB, -1)
    bc = np.bincount(r_bits.sum((1, 3)).view(-1).tolist())

    bc = 100. * (bc / bc.sum())
    xs = np.arange(len(bc))
    long_tail = xs[np.cumsum(bc) > 99][0]
    plt.figure(figsize=(10, 3.5))
    rect = patches.Rectangle((long_tail, 0), 200, 3, fill=True, color='r', alpha=0.05, zorder=0)
    plt.gca().add_patch(rect)
    plt.plot([long_tail, long_tail], [0, 2.8], color='r', linewidth=2, linestyle='--',
             zorder=1)
    plt.fill_between(xs, bc, zorder=2, color='cornflowerblue')
    plt.plot(xs, bc, '-k', linewidth=2)
    plt.text(111, 1.3, '99% of partial dot\nproducts (groups of 16)\nrequire fewer than 110\nterm pair multiplications.\nThe theoretical max\nis 16*7*7 = 784.', fontdict={'color': 'r'})
    plt.ylim(0, 2.6)
    plt.xlim(0, 165)
    plt.title('Term Pair Multiplications in Dot Products (Groups of 16)')
    plt.xlabel('Term Pair Multiplications')
    plt.ylabel('Frequency (%)')
    plt.savefig('figures/term-group-dist.pdf', dpi=300, bbox_inches='tight')
    plt.clf()
