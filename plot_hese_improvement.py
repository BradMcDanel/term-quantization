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


names = ['Binary', 'Booth radix-4', 'HESE']
real_colors = ['black', 'salmon', 'steelblue']
fake_colors = ['gray', 'lightsalmon', 'lightsteelblue']
funcs = [
    booth.num_binary_terms,
    booth.num_radix4_terms,
    booth.num_hese_terms,
]

binary_terms = booth.num_binary_terms(real_x, sf).view(-1).cpu().numpy().astype('i')
hese_terms = booth.num_hese_terms(real_x, sf).view(-1).cpu().numpy().astype('i')

term_hist = []
norm = len(hese_terms)
for i in range(6):
    bc = np.bincount(hese_terms[binary_terms==i], minlength=7)
    bc = 100. * (bc / norm)
    term_hist.append(bc)

term_hist = np.array(term_hist).transpose()

ps = []
ind = np.arange(6)
names = []
for i in ind:
    p = plt.bar(ind, term_hist[i])
    names.append('{} terms'.format(i))
    ps.append(p)

plt.legend(ps, names)
plt.xlabel('Number of Binary Terms')
plt.ylabel('Percentage of HESE Terms')
plt.savefig('figures/term_bars.pdf', dpi=300, bbox_inches='tight')
plt.clf()