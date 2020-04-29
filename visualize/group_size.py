import json

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
with open('results/resnet18-group-size-results.json', 'r') as fp:
    model_results = json.load(fp)

for key, value in list(model_results.items()):
    if key in ['16']:
        continue
    label = fancy_names[key]
    color = colors['{}'.format(key)]
    linestyle = line_styles[key]
    marker = marker_styles[key]
    terms, accs = [], []
    for term, acc in zip(value['avg_terms'], value['accs']):
        if  term >= 1 and term <= 3:
            terms.append(term)
            accs.append(acc)
    ax1.plot(terms, accs, '-o', linewidth=2.5, ms=6, color=color, marker=marker,
             linestyle=linestyle, markeredgecolor='k', markeredgewidth=0.8, label=label)

ax1.legend(loc='lower right', ncol=2)
ax1.set_title(r'Impact of Group Size $g$')
ax1.set_xlabel(r'Average Number of Terms $\alpha$')
ax1.set_ylabel('Top-1 Accuracy (%)')
plt.tight_layout()
plt.savefig('figures/group-size-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()
