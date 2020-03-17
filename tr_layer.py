import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

from util import AverageMeter

booth_cuda = load('booth_cuda', ['kernels/booth_cuda.cpp',
                  'kernels/booth_cuda_kernel.cu'], extra_cflags=['-O3'])


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def profile_tensor(w, err_tol):
    from itertools import product
    group_sizes = [16]
    term_factors = [0.5, 0.75, 0.875, 1, 1.125, 1.25, 1.375, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.0]
    # bit_settings = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
    # bit_settings = [6, 7, 8, 9, 10]
    bit_settings = [10]
    settings = product(bit_settings, group_sizes, term_factors)

    for weight_bits, group_size, term_factor in settings:
        num_terms = round(group_size * term_factor)
        max_wq = 2**(weight_bits - 1)
        w_sf = w.abs().max().item() / max_wq
        wq = booth_cuda.radix_2_mod(w, w_sf, weight_bits, group_size, num_terms)
        avg_err = (wq - w).abs().mean().item()
        if avg_err < err_tol:
            return weight_bits, group_size, num_terms
    
    print('Could not find setting below err_tol: ', err_tol)
    assert False

def mse_profile(hist, minv, maxv, bit_width):
    x = torch.linspace(minv, maxv, len(hist)).cuda()
    sfs = (1 / torch.linspace(1e-8, maxv, 2048)).tolist()
    errs = []
    for sf in sfs:
        max_xq = 2**(bit_width - 1)
        xq = linear_quantize_clamp(x, sf, -max_xq, max_xq)
        xh = linear_dequantize(xq, sf)
        err = (hist * (x - xh)**2).sum()
        errs.append(err)

    min_idx = torch.argmin(torch.Tensor(errs)).item()
    return sfs[min_idx]


class LinearQuantize(nn.Module):
    def __init__(self, data_bits):
        super(LinearQuantize, self).__init__()
        self.sf = 1
        self.num_bins = 2048
        self.minv = -6
        self.maxv = 6
        self.register_buffer('hist_bins', torch.Tensor(self.num_bins).zero_())
        self.tracking = True
        self.data_bits = data_bits
    
    def forward(self, x):
        max_xq = 2**(self.data_bits - 1)

        if self.tracking:
            self.hist_bins += torch.histc(x, self.num_bins, self.minv,
                                          self.maxv)
            return x

        xq = linear_quantize_clamp(x, self.sf, -max_xq, max_xq)
        return linear_dequantize(xq, self.sf)
    
    def finish_tracking(self):
        self.sf = mse_profile(self.hist_bins, self.minv, self.maxv,
                              self.data_bits)
        self.tracking = False
        

class TRConv2dLayer(nn.Module):
    def __init__(self, conv_layer, data_bits=8, weight_bits=8,
                 group_size=1, num_terms=8):
        super(TRConv2dLayer, self).__init__()
        device = conv_layer.weight.device
        self.input_quant = LinearQuantize(data_bits).to(device)
        self.group_size = group_size
        self.num_terms = num_terms
        self.data_bits = data_bits
        self.weight_bits = weight_bits
        w = conv_layer.weight
        max_wq = 2**(self.weight_bits - 1)
        self.w_sf = w.abs().max().item() / max_wq
        w = booth_cuda.radix_2_mod(w, self.w_sf, weight_bits, self.group_size,
                                   self.num_terms)
        conv_layer.weight = nn.Parameter(w)
        self.conv = conv_layer
    
    def forward(self, x):
        xq = self.input_quant(x)
        return self.conv(xq)
    
    def tracking(self, tracking):
        if tracking == False:
            self.input_quant.finish_tracking()
        else:
            self.input_quant.tracking = True