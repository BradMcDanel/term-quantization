from itertools import product
import math

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

booth_cuda = load(
    'booth_cuda', ['kernels/booth_cuda.cpp', 'kernels/booth_cuda_kernel.cu'], extra_cflags=['-O3'])

def binary_encode(number):
    char_number = bin(number).split('b')[1]
    char_number = char_number[::-1]
    exponents = []
    for i, bit in enumerate(char_number):
        if bit == '1':
            exponents.append(2**i)

    return exponents   

def quant(tensor, sf, num_exps, func_type):
    if func_type == 'binary':
        f = binary_encode
    elif func_type == 'radix-2':
        f = radix_2
    elif func_type == 'radix-2-hack':
        f = radix_2_hack
    elif func_type == 'radix-4':
        f = radix_4
    elif func_type == 'radix-8':
        f = radix_8

    shape = tensor.shape
    tensor = tensor.view(-1).tolist()
    truncated_tensor = []
    for number in tensor:
        trunc_number = f(int(number / sf))[-num_exps:]
        truncated_tensor.append(sum(trunc_number))
    
    truncated_tensor = torch.Tensor(truncated_tensor).view(*shape).cuda()
    truncated_tensor *= sf
    return truncated_tensor

def radix_2(number):
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

def radix_2_hack(number):
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
        
    # merging neighbors hack
    keep_exponents = []
    for i in range(0, len(exponents), 2):
        if exponents[i+1] == -(2*exponents[i]):
            keep_exponents.append(-exponents[i])
        else:
            keep_exponents.append(exponents[i])
            keep_exponents.append(exponents[i+1])

    return keep_exponents

def radix_2_hack2(number):
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = char_number 
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

def radix_2_hack3(number):
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = char_number 
    char_number = char_number[::-1]
    exponents = []
    i = 0
    while i < len(char_number):
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
            i += 1
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

        i += 1
        print(i)

    return exponents

def radix_4(number):
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = char_number + '0'
    char_number = '00' + char_number
    char_number = char_number[::-1]
    exponents = [] 
    bit_pos = 0
    for i in range(1, len(char_number) - 1, 2):
        b1 = char_number[i-1]
        b2 = char_number[i]
        b3 = char_number[i+1]
        if b3 == '0' and b2 == '0' and b1 == '0':
            pass
        elif b3 == '0' and b2 == '0' and b1 == '1':
            exponents.append(2**bit_pos)
        elif b3 == '0' and b2 == '1' and b1 == '0':
            exponents.append(2**bit_pos)
        elif b3 == '0' and b2 == '1' and b1 == '1':
            exponents.append(2**(bit_pos+1))
        elif b3 == '1' and b2 == '0' and b1 == '0':
            exponents.append(-2**(bit_pos+1))
        elif b3 == '1' and b2 == '0' and b1 == '1':
            exponents.append(-2**bit_pos)
        elif b3 == '1' and b2 == '1' and b1 == '0':
            exponents.append(-2**bit_pos)
        elif b3 == '1' and b2 == '1' and b1 == '1':
            pass

        bit_pos += 2

    return exponents

def radix_8(number):
    char_number = bin(number).split('b')[1]
    if bin(number)[0] == '-':
        sign = -1
    else:
        sign = 1
    char_number = char_number + '0'
    if char_number[-1] == '0':
        char_number = '0'*5 + char_number
    else:
        char_number = '1'*5 + char_number

    char_number = char_number[::-1]
    exponents = [] 
    bit_pos = 0
    for i in range(1, len(char_number) - 2, 3):
        b1 = char_number[i-1]
        b2 = char_number[i]
        b3 = char_number[i+1]
        b4 = char_number[i+2]
        if b4 == '0' and b3 == '0' and b2 == '0' and b1 == '0':
            pass
        elif b4 == '0' and b3 == '0' and b2 == '0' and b1 == '1':
            exponents.append(2**bit_pos)
        elif b4 == '0' and b3 == '0' and b2 == '1' and b1 == '0':
            exponents.append(2**bit_pos)
        elif b4 == '0' and b3 == '0' and b2 == '1' and b1 == '1':
            exponents.append(2**(bit_pos+1))
        elif b4 == '0' and b3 == '1' and b2 == '0' and b1 == '0':
            exponents.append(2**(bit_pos+1))
        elif b4 == '0' and b3 == '1' and b2 == '0' and b1 == '1':
            exponents.append(2**bit_pos)
            exponents.append(2**(bit_pos+1))
        elif b4 == '0' and b3 == '1' and b2 == '1' and b1 == '0':
            exponents.append(2**bit_pos)
            exponents.append(2**(bit_pos+1))
        elif b4 == '0' and b3 == '1' and b2 == '1' and b1 == '1':
            exponents.append(2**(bit_pos+2))
        elif b4 == '1' and b3 == '0' and b2 == '0' and b1 == '0':
            exponents.append(-2**(bit_pos+2))
        elif b4 == '1' and b3 == '0' and b2 == '0' and b1 == '1':
            exponents.append(-2**bit_pos)
            exponents.append(-2**(bit_pos+1))
        elif b4 == '1' and b3 == '0' and b2 == '1' and b1 == '0':
            exponents.append(-2**bit_pos)
            exponents.append(-2**(bit_pos+1))
        elif b4 == '1' and b3 == '0' and b2 == '1' and b1 == '1':
            exponents.append(-2**(bit_pos+1))
        elif b4 == '1' and b3 == '1' and b2 == '0' and b1 == '0':
            exponents.append(-2**(bit_pos+1))
        elif b4 == '1' and b3 == '1' and b2 == '0' and b1 == '1':
            exponents.append(-2**bit_pos)
        elif b4 == '1' and b3 == '1' and b2 == '1' and b1 == '0':
            exponents.append(-2**bit_pos)
        elif b4 == '1' and b3 == '1' and b2 == '1' and b1 == '1':
            pass

        bit_pos += 3

    return exponents

def pad_torch_mat(xs, pad_value=0, min_dim=8):
    for i in range(len(xs)):
        ilen = len(xs[i])
        diff = min_dim - ilen
        if diff > 0:
            xs[i].extend([pad_value]*diff)

    return torch.Tensor(xs)

def min_power_rep(max_exp, max_terms):
    from itertools import product
    sign = lambda x: (1, -1)[x <= 0]
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
            if value >= 0 and value < num_values:
                if values[value] is None:
                    values[value] = list(terms)[::-1]
                elif len(terms) > len(values[value]):
                    pass
                else:
                    exp_sum = sum([2**(abs(term)-1) for term in terms])
                    curr_exp_sum = sum([2**(abs(term)-1)for term in values[value]])
                    if exp_sum > curr_exp_sum:
                        values[value] = list(terms)[::-1]


    return values


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
    def __init__(self, sf, group_size, num_exps):
        super(BoothGroupQuant, self).__init__()
        self.sf = sf
        self.group_size = group_size
        self.num_exps = num_exps

    def forward(self, x):
        return booth_cuda.group(x, self.sf, self.group_size, self.num_exps)

class Radix2ModGroup(nn.Module):
    def __init__(self, sf, group_size, num_exps):
        super(Radix2ModGroup, self).__init__()
        self.sf = sf
        self.group_size = group_size
        self.num_exps = num_exps

    def forward(self, x):
        return booth_cuda.radix_2_mod(x, self.sf, self.group_size, self.num_exps)

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
    values = min_power_rep(8, 6)
    # values = pad_torch_mat(values, min_dim=6).int().cuda()
    for i in range(128):
        binary_values = binary_encode(i)
        radix_2_values = radix_2(i)
        radix_4_values = radix_4(i)
        radix_8_values = radix_8(i)
        search = values[384+i]
        print('{}: {}, {}, {}, {}, {}'.format(i, binary_values, radix_2_values, radix_4_values,
                                              radix_8_values, search))
        print('{}: {}, {}, {}, {}, {}'.format(i, len(binary_values), len(radix_2_values),
                                              len(radix_4_values), len(radix_8_values), len(search)))
    assert False

    sf = 2**-6
    assert False
    print((sf * (x[1,0,0,:10] / sf).int().float()).tolist())
    for i in range(1, 8):
        xcmp = booth_cuda.weight_single(x, values, sf, i)
        print(xcmp[1,0,0,:10].tolist())
