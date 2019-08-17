import torch
import numpy as np

import models
import booth
import util
plt = util.import_plt_settings(local_display=True)

def mac_term_pairs(x, w):
    max_x = x.max().int().item()
    max_w = w.max().int().item()
    term_pair_bins = [0] * (max_w * max_x + 1)
    M, N, K = w.shape[0], x.shape[1], w.shape[1]
    for m in range(M):
        if m == 10: break
        for n in range(N):
            for k in range(K):
                num_term_pairs = (w[m, k] * x[k, n]).int().item()
                term_pair_bins[num_term_pairs] += 1

    return term_pair_bins

def tiled_mac_term_pairs(x, w, tile_size=1):
    max_x = x.max().int().item()
    max_w = w.max().int().item()
    # term_pair_bins = [0] * (max_w * max_x + 1)
    term_pair_values = []
    M, N, K = w.shape[0], x.shape[1], w.shape[1]
    for m in range(M):
        if m == 10: break
        for n in range(N):
            for k in range(K):
                num_term_pairs = (w[m, k] * x[k, n]).int().item()
                # term_pair_bins[num_term_pairs] += 1
                term_pair_values.append(num_term_pairs)

    return term_pair_bins

model = models.shiftnet19(pretrained=True)
layer = util.get_layers(model, [torch.nn.Conv2d])[10]
real_w = layer.weight.data.view(-1).cuda()
WF, WC = layer.weight.shape[:2]
C, W, H = 256, 28, 28
real_x = torch.load('data/layer_10_data.pt')[:C*W*H].cuda()

sf = 2**-8
x_ticks = np.array([0, 1, 2, 3, 4, 6, 8, 9, 12, 16])

# baseline
# x = (torch.clamp(torch.floor((real_x / sf) + 0.5), -128, 128)).int().tolist()
# w = (torch.clamp(torch.floor((real_w / sf) + 0.5), -128, 128)).int().tolist()
# data_terms = [min(len(booth.radix_2_hack2(xi)), 4) for xi in x]
# data_terms = torch.Tensor(data_terms).view(C, W*H)
# weight_terms = [min(len(booth.radix_2_hack2(wi)), 4) for wi in w]
# weight_terms = torch.Tensor(weight_terms).view(WF, WC)
# baseline = mac_term_pairs(data_terms, weight_terms)
# baseline = np.array([baseline[i] for i in x_ticks])
# baseline = 100 * (baseline / np.sum(baseline))

# term revealing
x = torch.clamp(torch.floor((real_x / sf) + 0.5), -128, 128) * sf
x = x.view(1, C, W, H).cuda()
x = (booth.booth_cuda.radix_2_mod(x, sf, 8, 12) / sf).view(-1).int().tolist()
w = torch.clamp(torch.floor((real_w / sf) + 0.5), -128, 128) * sf
w = w.view(WF, WC, 1, 1).cuda()
w = (booth.booth_cuda.radix_2_mod(w, sf, 8, 12) / sf).view(-1).int().tolist()
data_terms = [min(len(booth.radix_2_hack2(xi)), 4) for xi in x]
data_terms = torch.Tensor(data_terms).view(C, W*H)
weight_terms = [min(len(booth.radix_2_hack2(wi)), 4) for wi in w]
weight_terms = torch.Tensor(weight_terms).view(WF, WC)
term_reveal = mac_term_pairs(data_terms, weight_terms)
term_reveal = np.array([term_reveal[i] for i in x_ticks])
term_reveal = 100 * (term_reveal / np.sum(term_reveal))


plt.figure(figsize=(5.5, 4.5))
bar_width = 0.4
names = ['Baseline', 'Term Revealing (8, 12)']
colors = [util.lighten_color('b', 0.8), util.lighten_color('r', 0.8)]
terms = [baseline, term_reveal]
for i in range(len(terms)):
    pos = np.arange(len(x_ticks)) + i*bar_width
    plt.bar(pos, terms[i], width=bar_width, edgecolor='k', color=colors[i], label=names[i])

plt.title('Frequency of Term Pair Occurences')
plt.xlabel('Number of Terms Pairs per MAC')
plt.ylabel('Frequency (%)')
plt.xticks(np.arange(len(x_ticks)) + 0.5*bar_width, x_ticks)
plt.legend(loc=0)
plt.savefig('figures/term-pair-histogram.pdf', dpi=300, bbox_inches='tight')
plt.clf()