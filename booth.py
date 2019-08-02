from itertools import product
import math

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

booth_cuda = load(
    'booth_cuda', ['kernels/booth_cuda.cpp', 'kernels/booth_cuda_kernel.cu'], extra_cflags=['-O3'])

def twos_complement(input_value, num_bits):
    '''Calculates a two's complement integer from the given input value's bits'''
    mask = 2**(num_bits - 1)
    return -(input_value & mask) + (input_value & ~mask)

def encode(number):
    # number = twos_complement(number, bits)
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = '0' + char_number + '0'
    char_number = char_number[::-1]
    exponents = []
    for i in range(len(char_number) - 1):
        b1 = char_number[i]
        b2 = char_number[i+1]
        if b1 == b2:
            continue
        if b1 == '0':
            exponents.append(-sign*2**i)
        else:
            exponents.append(sign*2**i)

    return exponents

def lossy_encode(number, num_exps=8):
    return sum(encode(number)[-num_exps:])

def pad_torch_mat(xs, pad_value=0, min_dim=8):
    for i in range(len(xs)):
        ilen = len(xs[i])
        diff = min_dim - ilen
        if diff > 0:
            xs[i].extend([pad_value]*diff)

    return torch.Tensor(xs)

def min_power_rep(max_exp, max_terms):
    from itertools import product
    sign = lambda x: (1, -1)[x < 0]
    exp_values = [-v for v in range(1, max_exp + 1)] + [v for v in range(1, max_exp + 1)]
    exp_values = sorted(exp_values, key=lambda x: abs(x))
    num_values = 3*2**max_exp 
    values = [None] * num_values
    mid_point = int(len(values) / 2)
    values[mid_point] = [0]
    for num_terms in range(1, max_terms + 1):
        for terms in product(exp_values, repeat=num_terms):
            rep_value = sum([sign(term)*2**(abs(term) - 1) for term in terms])
            value = rep_value + mid_point
            if value < num_values and values[value] is None:
                values[value] = list(terms)[::-1]

    return values


def two_powers(min_exp, max_exp):
    max_val = 2*2**(max_exp-1)
    values, value_powers = [], [None for _ in range(max_val*2+1)]
    for i in range(min_exp, max_exp):
        for j in range(min_exp, max_exp):
            v = 2**i + 2**j
            value_powers[v+max_val] = [i, j]
            values.append(v)

            v = -2**i + 2**j
            value_powers[v+max_val] = [-i, j]
            values.append(v)

            v = 2**i + -2**j
            value_powers[v+max_val] = [i, -j]
            values.append(v)

            v = -2**i + -2**j
            value_powers[v+max_val] = [-i, -j]
            values.append(v)

    values = sorted(list(set(values)))
    return values, value_powers

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)

def get_powers(number, value_powers, max_exp, num_powers=2):
    if number == 0: return [0]
    if is_power2(abs(number)): return [int(math.copysign(math.log2(abs(number)), number))]
    max_val = num_powers*2**(max_exp-1)
    return value_powers[number + max_val]

def three_powers(min_exp, max_exp):
    max_val = 3*2**(max_exp-1)
    values = []
    values, value_powers = [], [None for _ in range(max_val*2+1)]
    for i in range(min_exp, max_exp):
        for j in range(min_exp, max_exp):
            for k in range(min_exp, max_exp):
                v = 2**i + 2**j + 2**k
                value_powers[v+max_val] = [i, j, k]
                values.append(v)

                v = -2**i + 2**j + 2**k
                value_powers[v+max_val] = [-i, j, k]
                values.append(v)

                v = 2**i + 2**j - 2**k
                value_powers[v+max_val] = [i, j, -k]
                values.append(v)

                v = -2**i + 2**j - 2**k
                value_powers[v+max_val] = [-i, j, -k]
                values.append(v)

                v = 2**i - 2**j + 2**k
                value_powers[v+max_val] = [i, -j, k]
                values.append(v)

                v = 2**i - 2**j - 2**k
                value_powers[v+max_val] = [i, -j, -k]
                values.append(v)

                v = -2**i - 2**j - 2**k
                value_powers[v+max_val] = [-i, -j, -k]
                values.append(v)

    values = values + [-v for v in values] + [0]
    values = sorted(list(set(values)))
    return values, value_powers

def booth_quant_map(booth_values, delta, bits):
    booth_values = delta * torch.Tensor(booth_values).float().cuda()
    quant_map = delta * torch.arange(-2**(bits-1), 2**(bits-1)).float().cuda() 
    booth_quant_map = []
    for val in quant_map:
        idx = (val - booth_values).abs().argmin().item()
        booth_quant_map.append(booth_values[idx])
    
    booth_quant_map = torch.Tensor(booth_quant_map)
    return booth_quant_map

class BoothGroupQuant(nn.Module):
    def __init__(self, sf, group_size, num_max_exp):
        super(BoothGroupQuant, self).__init__()
        self.sf = sf
        self.group_size = group_size
        self.num_max_exp = num_max_exp

    def forward(self, x):
        return booth_cuda.forward(x, self.sf, self.group_size, self.num_max_exp)

class BoothQuant(nn.Module):
    def __init__(self, sf, num_exps=8):
        super(BoothQuant, self).__init__()
        self.sf = sf
        self.num_exps = num_exps
    
    def forward(self, x):
        return booth_cuda.single(x, self.sf, self.num_exps)

def pow2_quant(x, pow2_values):
    B, C, W, H = x.shape
    x = x.view(-1)
    idxs = (x.unsqueeze(0) - pow2_values.unsqueeze(1)).abs().min(dim=0)[1]
    x = pow2_values[idxs].view(B, C, W, H)
    return x

def weight_single_quant(x, sf, terms, num_keep_terms):
    return booth_cuda.weight_single(x, terms, sf, num_keep_terms)

def weight_group_quant(x, sf, terms, group_size, num_keep_terms):
    B, C, W, H = x.shape
    if W != 1 or H != 1:
        x = x.view(B, C*W*H, 1, 1)
    xq = booth_cuda.weight_group(x, terms, sf, group_size, num_keep_terms)
    return xq.view(B, C, W, H)

class Pow2Quant(nn.Module):
    def __init__(self, pow2_values):
        super(Pow2Quant, self).__init__()
        self.pow2_values = pow2_values
    
    def forward(self, x):
        return pow2_quant(x, self.pow2_values)
       

if __name__=='__main__':
    import torch
    from torch.autograd import Variable
    x = Variable(torch.Tensor(2, 8, 16, 16).uniform_(-1, 1).double().cuda())
    assert False
    values = min_power_rep(8, 6)
    values = pad_torch_mat(values, min_dim=6).int().cuda()

    sf = 2**-6
    assert False
    print((sf * (x[1,0,0,:10] / sf).int().float()).tolist())
    for i in range(1, 8):
        xcmp = booth_cuda.weight_single(x, values, sf, i)
        print(xcmp[1,0,0,:10].tolist())
