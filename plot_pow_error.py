import torch

import booth
import util
plt = util.import_plt_settings(local_display=True)

# values = booth.min_power_rep(8, 6)
# values = booth.pad_torch_mat(values, min_dim=6).int().cuda()

real_x = torch.load('data/layer_10_data.pt')[:10000].cuda()
# x = torch.Tensor(10000, 1, 1, 1).normal_(0, 1).cuda()
# x = torch.Tensor(10000, 1, 1, 1).uniform_(0, 6).cuda()
sf = 2**-7
# sf = 2**-4

# clamp x
# x[x < 0] = 0
# x[x > 128*sf] = 128*sf
num_exps = list(range(1, 7))
names = ['Binary', 'Booth (Radix-2)', 'Booth (Mod. Radix-2)',
         'Booth (Radix-4)', 'Booth (Radix-8)']
colors = ['']
funcs = [
    lambda x, ne: booth.quant(x, sf, ne, 'binary'),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-2'),
    lambda x, ne: booth.booth_cuda.radix_2_mod(x.view(1,-1, 1, 1), sf, 1, ne).view(-1),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-4'),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-8'),
]

errors = []
for func in funcs:
    error_row = []
    for ne in num_exps:
        xq = func(real_x, ne)
        error = (real_x - xq).abs().sum().item() / real_x.numel()
        error_row.append(error)
    errors.append(error_row)

for name, error in zip(names, errors):
    plt.plot(num_exps, error, '-o',  linewidth=2.5, ms=8,
             markeredgecolor='k', markeredgewidth=0.8, label=name)

plt.xlabel('Max Number of Terms per Expression')
plt.ylabel('Average Truncation Error (log)')
plt.yscale('log')
plt.legend(loc=0)
plt.xticks(num_exps)
plt.tight_layout()
plt.savefig('figures/freq/representation_error.pdf', dpi=300)
plt.clf()


assert False






plt.savefig('figures/freq/pow_error.png')
plt.clf()

sf = 2**-7
# groups = [1, 2, 4, 8]
groups = [1, 4, 8, 16]
num_exps = list(range(1, 7))
colors = ['r', 'g', 'b', 'k']
errors = []
for group in groups:
    error_row = []
    for ne in num_exps:
        layer = booth.BoothGroupQuant(sf, group, group*ne).cuda()
        xq = layer(x)
        error = (x - xq).abs().sum().item() / x.numel()
        error_row.append(error)
    errors.append(error_row)

for i, group in enumerate(groups):
    if i == 0:
        plt.plot(num_exps, errors[i], '-o', color=colors[i], linewidth=2.5, ms=8,
                 markeredgecolor='k', markeredgewidth=0.8, label='No Grouping')
    else:
        plt.plot(num_exps, errors[i], '-o', color=colors[i], linewidth=2.5, ms=8,
                 markeredgecolor='k', markeredgewidth=0.8, label='Group Size of {}'.format(group))


plt.xlabel('Average Number of Terms per Expression')
plt.ylabel('Average Quantization Error')
plt.legend(loc=0)
plt.xticks(num_exps)
plt.tight_layout()
plt.savefig('figures/freq/data_error.pdf', dpi=300)
plt.clf()

# weight combining
w = torch.Tensor(10, 128, 32, 32).normal_(0, 0.2).cuda()
sf = 2**-7
groups = [1, 2, 4, 8]
num_exps = list(range(1, 8))
errors = []
for group in groups:
    error_row = []
    for ne in num_exps:
        layer = booth.BoothGroupQuant(sf, group, group*ne).cuda()
        wq = layer(w)
        error = (w - wq).abs().sum().item() / x.numel()
        error_row.append(error)
    errors.append(error_row)

for i, group in enumerate(groups):
    plt.plot(num_exps, errors[i], '-o', linewidth=2.5, label='G-{}'.format(group))
plt.xlabel('Number of Weight Terms')
plt.ylabel('Average L1 Error |w - w_quant|')
plt.legend(loc=0)
plt.xticks(num_exps)
plt.tight_layout()
plt.savefig('figures/freq/weight_error.png', dpi=300)
plt.clf()


# comparing weight and data quant
