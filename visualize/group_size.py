import json

from matplotlib.patches import Rectangle

import visualize
plt = visualize.import_plt_settings(local_display=True)

fancy_names = {
    '1': r'$g$=$1$',
    '2': r'$g$=$2$',
    '4': r'$g$=$4$',
    '8': r'$g$=$8$',
    '16': r'$g$=$16$',
    '32': r'$g$=$32$',
}

colors = {
    '1': '#bd0026',
    '2': '#f03b20',
    '4': '#fd8d3c',
    '8': '#fecc5c',
    '16': '#edddc2',
    '32': '#edddc2',
}

line_styles = {
    '1': '-',
    '2': '-',
    '4': '-',
    '8': '-',
    '16': '-',
    '32': '-',
}

marker_styles = {
    '1': 'o',
    '2': '^',
    '8': 'v',
    '32': 's',
}

fig, ax1 = plt.subplots(figsize=(5, 4.5))
ax2 = fig.add_axes([0.54, 0.42, 0.4, 0.34])
with open('results/shiftnet19-group-results.txt', 'r') as fp:
    model_results = json.load(fp)

for key, value in list(model_results.items()):
    if key in ['16']:
        continue
    label = fancy_names[key]
    color = colors['{}'.format(key)]
    linestyle = line_styles[key]
    marker = marker_styles[key]
    terms, accs = [], []
    for term, acc in zip(value['avg_terms'], value['acc']):
        if  term >= 1 and term <= 3:
            terms.append(term)
            accs.append(acc)
    ax1.plot(terms, accs, '-o', linewidth=2.5, ms=6, color=color, marker=marker,
             linestyle=linestyle, markeredgecolor='k', markeredgewidth=0.8, label=label)
    ax2.plot(terms, accs, '-o', linewidth=2.5, ms=6, color=color, marker=marker,
             linestyle=linestyle, markeredgecolor='k', markeredgewidth=0.8)

ax1.add_patch(Rectangle((0.95, 64), 2.1 - 0.96, 3.5, alpha=1, facecolor='none',
                        edgecolor='k', zorder=10000))
ax1.legend(loc='lower right', ncol=2)
ax1.set_title(r'Impact of Group Size $g$')
ax1.set_xlabel(r'Average Number of Terms $\alpha$')
ax1.set_ylabel('ImageNet Top-1 Accuracy')
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.set_xticks([1.0, 1.25, 1.5, 1.75, 2.0])
ax2.set_yticks([65, 66])
ax2.set_xlim(0.96, 2.1)
ax2.set_ylim(64, 67)
plt.tight_layout()
plt.savefig('../figures/group-size-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
