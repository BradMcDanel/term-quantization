import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

tr_cuda = load('tr_cuda', ['kernels/tr_cuda.cpp', 'kernels/tr_cuda_kernel.cu'])

def mse_profile(hist, minv, maxv, bit_width, terms):
    x = torch.linspace(minv, maxv, len(hist)).cuda()
    sfs = torch.linspace(1e-8, maxv, 2048).tolist()
    errs = []
    for sf in sfs:
        xh = tr_cuda.tr(x.view(-1, 1, 1, 1), sf, bit_width, 1, terms)
        xh = xh.view(-1)
        err = (hist * (x - xh)**2).sum()
        errs.append(err)

    min_idx = torch.argmin(torch.Tensor(errs)).item()
    return sfs[min_idx]

class LinearQuantize(nn.Module):
    def __init__(self, data_bits, data_terms):
        super(LinearQuantize, self).__init__()
        self.sf = 1
        self.num_bins = 8192
        self.minv = -50
        self.maxv = 50
        self.register_buffer('hist_bins', torch.Tensor(self.num_bins).zero_())
        self.tracking = True
        self.data_bits = data_bits
        self.data_terms = data_terms

    def forward(self, x):
        if self.tracking:
            self.hist_bins += torch.histc(x, self.num_bins, self.minv,
                                          self.maxv)
            return x

        return tr_cuda.tr(x, self.sf, self.data_bits, 1, self.data_terms)

    def finish_tracking(self):
        self.sf = mse_profile(self.hist_bins, self.minv, self.maxv,
                              self.data_bits, self.data_terms)
        self.tracking = False

class TRConv2dLayer(nn.Module):
    def __init__(self, conv_layer, data_bits=8, data_terms=4, weight_bits=8,
                 group_size=1, num_terms=8):
        super(TRConv2dLayer, self).__init__()
        device = conv_layer.weight.device
        self.data_bits = data_bits
        self.data_terms = data_terms
        self.input_quant = LinearQuantize(data_bits, data_terms).to(device)
        self.group_size = group_size
        self.num_terms = num_terms
        self.weight_bits = weight_bits
        w = conv_layer.weight
        max_wq = 2**(self.weight_bits - 1)
        self.w_sf = w.abs().max().item() / max_wq
        w = tr_cuda.tr(w, self.w_sf, weight_bits, self.group_size, self.num_terms)
        conv_layer.weight = nn.Parameter(w)
        self.conv = conv_layer

    def forward(self, x):
        xq = self.input_quant(x)
        return self.conv(xq)

    def tracking(self, tracking):
        if not tracking:
            self.input_quant.finish_tracking()
        else:
            self.input_quant.tracking = True
