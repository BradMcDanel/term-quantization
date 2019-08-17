import torch
import numpy as np

import models
import booth
import util
plt = util.import_plt_settings(local_display=True)

model = models.shiftnet19(pretrained=True)
layer = util.get_layers(model, [torch.nn.Conv2d])[9]
real_w = layer.weight.data.view(-1).cuda()
real_x = torch.load('data/layer_10_data.pt')[:10000].cuda()

sf = 2**-8
x = (torch.clamp(torch.floor((real_x / sf) + 0.5), -128, 128)).int().tolist()
data_terms = [len(booth.radix_2_hack2(xi)) for xi in x]
data_term_count = np.array([0, 0, 0, 0, 0, 0])
for dt in data_terms:
    data_term_count[dt] += 1
data_term_count = 100 * (data_term_count / len(data_terms))

w = (torch.clamp(torch.floor((real_w / sf) + 0.5), -128, 128)).int().tolist()
weight_terms = [len(booth.radix_2_hack2(wi)) for wi in w]
weight_term_count = np.array([0, 0, 0, 0, 0, 0])
for wt in weight_terms:
    weight_term_count[wt] += 1
weight_term_count = 100 * (weight_term_count / len(weight_terms))


plt.figure(figsize=(5.5, 4.5))
bar_width = 0.4
names = ['Weights', 'Activations']
colors = ['#ABDDA4', '#FDB863']
x_ticks = np.array([0, 1, 2, 3, 4, 5])
terms = [weight_term_count, data_term_count]
for i in range(len(terms)):
    pos = x_ticks + i*bar_width
    plt.bar(pos, terms[i], width=bar_width, edgecolor='k', color=colors[i], label=names[i])

plt.title('(a) Histogram of Number of Terms')
plt.xlabel('Number of Terms')
plt.ylabel('Frequency (%)')
plt.xticks(x_ticks + 0.5*bar_width, x_ticks)
plt.legend(loc=0)
plt.savefig('figures/term-histogram.pdf', dpi=300, bbox_inches='tight')
plt.clf()