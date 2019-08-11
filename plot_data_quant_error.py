import torch

import booth
import util
plt = util.import_plt_settings(local_display=True)


# data combining
# x = torch.Tensor(10, 128, 32, 32).normal_(0, 1.2).cuda()
# x = torch.Tensor(10, 128, 32, 32).uniform_(-1, 1).cuda()
x = torch.load('layer_1_data.pt')[:10000]
# x[x < 0] = 0
assert False

x_flat = x.view(-1).tolist()
plt.hist(x_flat, bins=100)
plt.savefig('figures/freq/x_dist.png')
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
