import torch
import numpy as np

import booth
import util
plt = util.import_plt_settings(local_display=True)

N = 50000
sf = 2**-7
real_x = torch.load('data/layer_10_data.pt')[:N].cuda()
real_x = sf * torch.clamp(torch.floor((real_x / sf) + 0.5), -128, 128) 
fake_x = torch.Tensor(N).uniform_(-1, 1)
fake_x = sf * torch.clamp(torch.floor((fake_x / sf) + 0.5), -128, 128) 


names = ['Binary', 'Radix-4', 'HESE']
real_colors = ['black', 'salmon', 'steelblue']
fake_colors = ['gray', 'lightsalmon', 'lightsteelblue']
funcs = [
    booth.num_binary_terms,
    booth.num_radix4_terms,
    booth.num_hese_terms,
]

all_real_terms = []
all_fake_terms = []
for func in funcs:
    error_row = []

    # fake
    terms = func(fake_x, sf).view(-1).cpu().numpy().astype('i')
    terms = np.bincount(terms, minlength=8).tolist()
    all_fake_terms.append(terms)

    # real
    terms = func(real_x, sf).view(-1).cpu().numpy().astype('i')
    terms = np.bincount(terms, minlength=8).tolist()
    all_real_terms.append(terms)


plt.figure(figsize=(5, 5))

x = np.arange(len(all_real_terms[2]))
y = 100. * np.cumsum(all_real_terms[2]) / np.sum(all_real_terms[2])
plt.plot(x, y, '-o', linewidth=2.5, ms=7, color=real_colors[2],
         markeredgecolor='k', markeredgewidth=0.6, label=names[2] + ' (data)')

x = np.arange(len(all_fake_terms[2]))
y = 100. * np.cumsum(all_fake_terms[2]) / np.sum(all_fake_terms[2])
plt.plot(x, y, '--o', linewidth=2.5, ms=7, color=fake_colors[2],
         markeredgecolor='k', markeredgewidth=0.6, label=names[2]+ ' (unif)')

x = np.arange(len(all_real_terms[1]))
y = 100. * np.cumsum(all_real_terms[1]) / np.sum(all_real_terms[1])
plt.plot(x, y, '-^', linewidth=2.5, ms=7, color=real_colors[1],
         markeredgecolor='k', markeredgewidth=0.6, label=names[1] + ' (data)')

x = np.arange(len(all_fake_terms[1]))
y = 100. * np.cumsum(all_fake_terms[1]) / np.sum(all_fake_terms[1])
plt.plot(x, y, '--^', linewidth=2.5, ms=7, color=fake_colors[1],
         markeredgecolor='k', markeredgewidth=0.6, label=names[1]+ ' (unif)')


x = np.arange(len(all_real_terms[0]))
y = 100. * np.cumsum(all_real_terms[0]) / np.sum(all_real_terms[0])
plt.plot(x, y, '-v', linewidth=2.5, ms=7, color=real_colors[0],
         markeredgecolor='k', markeredgewidth=0.6, label=names[0] + ' (data)')

x = np.arange(len(all_fake_terms[0]))
y = 100. * np.cumsum(all_fake_terms[0]) / np.sum(all_fake_terms[0])
plt.plot(x, y, '--v', linewidth=2.5, ms=7, color=fake_colors[0],
         markeredgecolor='k', markeredgewidth=0.6, label=names[0] + ' (unif)')


plt.title('(c) Encoding Comparison')
plt.xlabel('Number of Terms')
plt.ylabel('Cumulative % of Values')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('figures/term-cdf.pdf', dpi=300, bbox_inches='tight')
plt.clf()