from itertools import product

import torch
import torch.nn as nn

def booth_two_values(min_exp, max_exp, signed=True):
    values = []
    for i in range(min_exp, max_exp):
        for j in range(min_exp, max_exp):
            values.append(2**i + 2**j)
            if signed:
                values.append(-2**i + 2**j)
                values.append(2**i - 2**j)
                values.append(-2**i - 2**j)
    values = values + [-v for v in values] + [0]
    values = sorted(list(set(values)))
    return values

def booth_three_values(min_exp, max_exp, signed=True):
    values = []
    for i in range(min_exp, max_exp):
        for j in range(min_exp, max_exp):
            for k in range(min_exp, max_exp):
                values.append(2**i + 2**j + 2**k)
                if signed:
                    values.append(-2**i + 2**j + 2**k)
                    values.append(2**i + 2**j - 2**k)
                    values.append(-2**i + 2**j - 2**k)
                    values.append(2**i - 2**j + 2**k)
                    values.append(2**i - 2**j - 2**k)
                    values.append(-2**i - 2**j - 2**k)
    values = values + [-v for v in values] + [0]
    values = sorted(list(set(values)))
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


def booth_quant(x, booth_values):
    B, C, W, H = x.shape
    x = x.view(-1)
    idxs = (x.unsqueeze(0) - booth_values.unsqueeze(1)).abs().min(dim=0)[1]
    x = booth_values[idxs].view(B, C, W, H)
    return x

class BoothQuant(nn.Module):
    def __init__(self, booth_values):
        super(BoothQuant, self).__init__()
        self.booth_values = booth_values
    
    def forward(self, x):
        return booth_quant(x, self.booth_values)
       

if __name__=='__main__':
    import torch
    import matplotlib
    import matplotlib.pyplot as plt
    SMALL_SIZE = 16
    MEDIUM_SIZE = 21
    BIGGER_SIZE = 25
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    values = booth_two_values(0, 7)
    values = 2**-4 * torch.Tensor(values).cuda()
    plt.plot(values.tolist(), linewidth=2.5)
    plt.xlabel('Quantization Index')
    plt.ylabel('Quantization Value')
    plt.tight_layout()
    plt.savefig('booth_dist.png', dpi=300)