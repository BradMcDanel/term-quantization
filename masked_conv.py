import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1):
        super(MaskedConv2d, self).__init__()
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)
        self.dilation = _make_pair(dilation)
        self.groups = groups
        self.bias = None
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = _make_pair(kernel_size)

        N = out_channels*in_channels*self.kernel_size[0]*self.kernel_size[1]
        n = self.kernel_size[0]*self.kernel_size[1]*out_channels
        self._weight = nn.Parameter(torch.Tensor(N))
        self._weight.data.normal_(0, math.sqrt(2. / n))
        self.register_buffer('_mask', torch.ones(N))

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
                    
    @property
    def weight(self):
        w = self.mask*self._weight
        return w.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

    @property
    def mask(self):
        return Variable(self._mask, requires_grad=False)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        return s.format(**self.__dict__)