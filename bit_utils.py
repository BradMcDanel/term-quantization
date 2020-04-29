from itertools import product
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hese(number):
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = char_number[::-1]
    exponents = []
    i = len(char_number) - 1
    while i >= 0:
        b0 = str(0 if i == 0 else char_number[i-1])
        b0 = str(b0)
        b1 = char_number[i]
        b2 = str(0 if i == len(char_number) - 1 else char_number[i+1])

        if b2 == '0' and b1 == '0' and b0 == '0':
            pass
        elif b2 == '0' and b1 == '0' and b0 == '1':
            pass
        elif b2 == '0' and b1 == '1' and b0 == '0':
            exponents.append(sign*2**i)
            i -= 1
        elif b2 == '0' and b1 == '1' and b0 == '1':
            exponents.append(sign*2**(i+1))
        elif b2 == '1' and b1 == '0' and b0 == '0':
            pass
        elif b2 == '1' and b1 == '0' and b0 == '1':
            pass
        elif b2 == '1' and b1 == '1' and b0 == '0':
            exponents.append(-sign*2**i)
        elif b2 == '1' and b1 == '1' and b0 == '1':
            pass

        i -= 1
    return exponents

def hese_bits(number, min_bits=8):
    exponents = hese(number)
    bit_repr = np.zeros(min_bits*2)
    for exponent in exponents:
        bit_idx = int(math.log2(abs(exponent)))
        bit_idx = bit_idx if exponent > 0 else bit_idx + min_bits
        bit_repr[bit_idx] = 1
    return bit_repr

def expand_hese_bits(W, sf, min_bits=9):
    W = torch.floor((W / sf) + 0.5).int()
    shape = W.shape
    W = W.view(-1)
    W = torch.Tensor([hese_bits(w, min_bits) for w in W]).cuda()
    W = W.view(*shape, -1)
    return W

def expand_binary_bits(W, sf, min_bits=9):
    def bin_func(w, min_bits):
        w = bin(abs(w))[2:].zfill(min_bits)
        w = [float(wi) for wi in w]
        return w
    W = torch.floor((W / sf) + 0.5).int()
    shape = W.shape
    W = W.view(-1)
    W = torch.Tensor([bin_func(w, min_bits) for w in W]).cuda()
    W = W.view(*shape, -1)
    return W