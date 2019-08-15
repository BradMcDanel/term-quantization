import json

import util
plt = util.import_plt_settings(local_display=True)

names = ['shiftnet19', 'vgg19_bn', 'resnet152']
fancy_names = {'fixed': 'Truncated', 'group': 'Term Reveal', 'alexnet': 'AlexNet', 'vgg19_bn': 'VGG-19',
               'shiftnet19': 'ShiftNet-19', 'resnet152': 'ResNet-152'}

colors = {
    'alexnet-fixed': 'b',
    'alexnet-group': util.lighten_color('b', 0.6),
    'vgg19_bn-fixed': 'r',
    'vgg19_bn-group': util.lighten_color('r', 0.6),
    'shiftnet19-fixed': 'g',
    'shiftnet19-group': util.lighten_color('g', 0.8),
    'resnet152-fixed': 'b',
    'resnet152-group': util.lighten_color('b', 0.4),
}

line_styles = {
    'alexnet-fixed': '-',
    'alexnet-group': '--',
    'vgg19_bn-fixed': '-',
    'vgg19_bn-group': '--',
    'shiftnet19-fixed': '-',
    'shiftnet19-group': '--',
    'resnet152-fixed': '-',
    'resnet152-group': '--',
}

plt.figure(figsize=(6.5, 4.5))
for name in names:
    with open('data/{}-results.txt'.format(name), 'r') as fp:
        model_results = json.load(fp)

    for key, value in model_results.items():
        label = '{} ({})'.format(fancy_names[name], fancy_names[key])
        color = colors['{}-{}'.format(name, key)]
        linestyle = line_styles['{}-{}'.format(name, key)]
        terms, accs = [], []
        for term, acc in zip(value['avg_terms'], value['acc']):
            if  term <= 3:
                terms.append(term)
                accs.append(acc)
        plt.plot(terms, accs, '-o', linewidth=2, ms=8, color=color,
                 linestyle=linestyle,
                 markeredgecolor='k', markeredgewidth=0.8, label=label)
    

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.legend(loc=0)
plt.tight_layout()
plt.title('Comparing Term Revealing to Truncation')
plt.xlabel(r'Average Number of Terms ($\alpha$)')
plt.ylabel('ImageNet Top-1 Accuracy')
plt.savefig('figures/model-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()