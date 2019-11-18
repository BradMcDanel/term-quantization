import torch

import booth
import util
plt = util.import_plt_settings(local_display=True)

real_x = torch.load('data/layer_10_data.pt')[:10000].cuda()
sf = 2**-7
x = sf * torch.clamp(torch.floor((real_x / sf) + 0.5), -128, 128) 

num_exps = list(range(1, 6))
names = ['Binary', 'Booth (Radix-2)', 'Booth (Radix-4)',
         'Booth (Radix-8)', 'HESE',]
colors = ['']
funcs = [
    lambda x, ne: booth.quant(x, sf, ne, 'binary'),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-2'),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-4'),
    lambda x, ne: booth.quant(x, sf, ne, 'radix-8'),
    lambda x, ne: booth.booth_cuda.radix_2_mod(x.view(1,-1, 1, 1), sf, 1, ne).view(-1),
]

errors = []
for func in funcs:
    error_row = []
    for ne in num_exps:
        xq = func(x, ne)
        error = (real_x - xq).abs().sum().item() / real_x.numel()
        error_row.append(error)
    errors.append(error_row)

# plt.figure(figsize=(5.5, 4.8))
for name, error in zip(names, errors):
    plt.plot(num_exps, error, '-o',  linewidth=2.5, ms=8,
             markeredgecolor='k', markeredgewidth=0.8, label=name)

plt.title('Error for Different Encodings')
plt.xlabel('Maximum Number of Terms')
plt.ylabel('Quantization Error (log)')
plt.yscale('log')
plt.legend(loc=0)
plt.xticks(num_exps)
plt.tight_layout()
plt.savefig('figures/representation-error.pdf', dpi=300, bbox_inches='tight')
plt.clf()