import torch
import torch.nn as nn
from efficientnet_pytorch.utils import Conv2dStaticSamePadding

import thop
import tr_layer

def tr_conv2d_ops(m, x, y):
    x = x[0]

    kernel_ops = torch.zeros(m.conv.weight.size()[2:]).numel()  # Kw x Kh

    # N x Cout x H x W x  (Cin x Kw x Kh + bias)
    total_ops = y.nelement() * (m.conv.in_channels // m.conv.groups * kernel_ops)

    # convert to term ops
    if m.group_size == 1:
        weight_terms = min(m.num_terms, m.weight_bits)
    else:
        weight_terms = m.num_terms
    data_terms = min(m.data_terms, m.data_bits)
    alpha = weight_terms / m.group_size
    total_ops = data_terms * alpha * total_ops

    if x.shape[1] > 3 and m.conv.groups == 1:
        m.conv.total_ops += torch.Tensor([int(total_ops)])

def tr_linear_ops(m, x, y):
    x = x[0]

    total_ops = y.nelement() * m.linear.in_features

    # convert to term ops
    if m.group_size == 1:
        weight_terms = min(m.num_terms, m.weight_bits)
    else:
        weight_terms = m.num_terms
    data_terms = min(m.data_terms, m.data_bits)
    alpha = weight_terms / m.group_size
    total_ops = data_terms * alpha * total_ops
    m.linear.total_ops += torch.Tensor([int(total_ops)])
    if m.group_size == 1:
        weight_bits = m.linear.weight.nelement() * m.weight_bits
    else:
        weight_bits = tr_layer.compute_compressed_hese(m.linear.weight, m.w_sf, m.weight_bits)
    m.linear.total_params += torch.Tensor([int(weight_bits)])

def tr_lstm_ops(m, x, y):
    x = x[0]

def get_model_ops(model, inputs):
    custom_ops = {
        tr_layer.TRConv2dLayer: tr_conv2d_ops,
        tr_layer.TRLinearLayer: tr_linear_ops,
        tr_layer.TRLSTMLayer: tr_lstm_ops,
        nn.Conv2d: thop.count_hooks.zero_ops,
        Conv2dStaticSamePadding: thop.count_hooks.zero_ops,
        nn.BatchNorm2d: thop.count_hooks.zero_ops,
        nn.Linear: thop.count_hooks.zero_ops,
        nn.AvgPool2d: thop.count_hooks.zero_ops,
        nn.AdaptiveAvgPool2d: thop.count_hooks.zero_ops
    }

    return thop.profile(model, inputs=inputs, custom_ops=custom_ops)
