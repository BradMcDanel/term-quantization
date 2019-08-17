import json

import util
plt = util.import_plt_settings(local_display=True)

# fancy_names = {
#     '1': r'Group Size ($g=1$)',
#     '2': r'Group Size ($g=2$)',
#     '4': r'Group Size ($g=4$)',
#     '8': r'Group Size ($g=8$)',
#     '16': r'Group Size ($g=16$)',
# }

fancy_names = {
    '1': r'$g=1$',
    '2': r'$g=2$',
    '4': r'$g=4$',
    '8': r'$g=8$',
    '16': r'$g=16$',
}

colors = {
    '1': '#bd0026',
    '2': '#f03b20',
    '4': '#fd8d3c',
    '8': '#fecc5c',
    '16': '#edddc2',
}

line_styles = {
    '1': '-',
    '2': '-',
    '4': '-',
    '8': '-',
    '16': '-',
}

plt.figure(figsize=(4.5, 4.5))
with open('data/vgg19_bn-group-results.txt', 'r') as fp:
    model_results = json.load(fp)

for key, value in model_results.items():
    label = fancy_names[key]
    color = colors['{}'.format(key)]
    linestyle = line_styles[key]
    terms, accs = [], []
    for term, acc in zip(value['avg_terms'], value['acc']):
        if  term <= 3:
            terms.append(term)
            accs.append(acc)
    plt.plot(terms, accs, '-o', linewidth=2, ms=8, color=color,
                linestyle=linestyle,
                markeredgecolor='k', markeredgewidth=0.8, label=label)


plt.legend(loc=0)
plt.tight_layout()
plt.title(r'(a) Impact of Group Size $g$')
plt.xticks(terms)
plt.xlabel(r'Average Number of Terms $\alpha$')
plt.ylabel('ImageNet Top-1 Accuracy')
plt.savefig('figures/group-size-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()