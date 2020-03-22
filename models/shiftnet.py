'''
Implementation based on:
"Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions"
(https://arxiv.org/pdf/1711.08141.pdf)
'''

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from torch.distributions import categorical

__all__ = ['ShiftNet', 'shiftnet19']

shift_cuda = load('shift_cuda', ['kernels/shift_cuda.cpp', 'kernels/shift_cuda_kernel.cu'])

class shift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift_dirs):
        ctx.save_for_backward(shift_dirs)
        return shift_cuda.forward(x, shift_dirs)

    @staticmethod
    def backward(ctx, grad_output):
        shift_dirs, = ctx.saved_tensors
        grad_output = shift_cuda.backward(grad_output, shift_dirs)

        return grad_output, None


class Shift(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(Shift, self).__init__()
        self.channels = in_channels
        self.kernel_size = kernel_size
        if kernel_size == 3:
            p = torch.Tensor([0.3, 0.4, 0.3])
        elif kernel_size == 5:
            p = torch.Tensor([0.1, 0.25, 0.3, 0.25, 0.1])
        elif kernel_size == 7:
            p = torch.Tensor([0.075, 0.1, 0.175, 0.3, 0.175, 0.1, 0.075])
        elif kernel_size == 9:
            p = torch.Tensor([0.05, 0.075, 0.1, 0.175, 0.2, 0.175, 0.1, 0.075, 0.05])
        else:
            raise RuntimeError('Unsupported kernel size')

        shift_t = categorical.Categorical(p).sample((in_channels, 2)) - (kernel_size // 2)
        self.register_buffer('shift_t', shift_t.int())

    def forward(self, x):
        if not x.is_cuda:
            print('Shift only supports GPU for now..')
            assert False

        return shift.apply(x, self.shift_t)

    def extra_repr(self):
        s = ('{channels}, kernel_size={kernel_size}')
        return s.format(**self.__dict__)

def shift_layer(in_channels, out_channels, kernel_size=3, stride=1):
    return [
        Shift(in_channels, kernel_size),
        nn.Conv2d(in_channels, out_channels, 1, stride, 0),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU()
    ]

def layer(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels, affine=True),
        nn.ReLU()
    ]

class ShiftNet(nn.Module):
    def __init__(self, settings, in_channels=3, num_classes=1000):
        super(ShiftNet, self).__init__()
        layers = []
        out_channels = settings[0][0]
        layers.extend(layer(in_channels, out_channels))
        in_channels = out_channels

        for out_channels, stride in settings:
            layers.extend(shift_layer(in_channels, out_channels, stride=stride))
            in_channels = out_channels

        self.feature_outchannels = out_channels
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_outchannels, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.feature_outchannels)
        x = self.classifier(x)
        return x

def shiftnet19(pretrained=False):
    settings = [
        [64, 1],
        [64, 1],
        [128, 2],
        [128, 1],
        [128, 1],
        [128, 1],
        [256, 2],
        [256, 1],
        [256, 1],
        [256, 1],
        [512, 2],
        [512, 1],
        [512, 1],
        [512, 1],
        [1024, 2],
        [1024, 1],
        [1024, 1],
    ]

    model = ShiftNet(settings)
    if pretrained:
        state_dict = torch.load('pth/shiftnet19-d372c4d2.pth')
        model.load_state_dict(state_dict)

    return model
