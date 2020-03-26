import numpy as np

import visualize
plt = visualize.import_plt_settings(local_display=True)

value_reveal = [22.22, 27.84, 40.41, 2.67]
term_reveal = [56.44, 65.76, 73.94, 76.77]
baseline_reveal = [56.51, 66.69, 74.11, 78.32]
width = 0.25
x_ticks = np.arange(len(value_reveal))


plt.figure(figsize=(5, 4.5))
plt.bar(x_ticks, value_reveal, label='Value Reveal', width=width, edgecolor='k',
        color='darkseagreen')
plt.bar(x_ticks+width, term_reveal, label='Term Reveal', width=width, hatch='/',
        edgecolor='k', color='tomato')
plt.bar(x_ticks+2*width, baseline_reveal, label='Baseline', width=width, hatch='//',
        edgecolor='k', color='grey')

plt.legend(loc=0)
plt.tight_layout()
plt.title('(a) Versus Value Revealing')
plt.ylim(top=100)
plt.ylabel('ImageNet Top-1 Accuracy')
plt.xticks(x_ticks+width, ['AlexNet', 'VGG-19', 'ResNet-151', 'ShiftNet-19'], fontsize=12)
plt.savefig('figures/value-reveal.pdf', bbox_inches='tight', dpi=300)
plt.clf()