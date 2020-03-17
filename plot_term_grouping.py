import json

import util
plt = util.import_plt_settings(local_display=True)
import matplotlib
matplotlib.rc('legend', fontsize=12)

names = ['shiftnet19', 'vgg19_bn', 'resnet152']
fancy_names = {'fixed': 'Truncate', 'group': 'TR', 'alexnet': 'AlexNet', 'vgg19_bn': 'VGG-19',
               'shiftnet19': 'ShiftNet-19', 'resnet152': 'ResNet-152'}

colors = {
    'alexnet-fixed':  util.lighten_color('b', 0.6),
    'alexnet-group': 'b',
    'vgg19_bn-fixed': util.lighten_color('r', 0.6),
    'vgg19_bn-group': 'r',
    'shiftnet19-fixed': util.lighten_color('g', 0.8),
    'shiftnet19-group': 'g',
    'resnet152-fixed': util.lighten_color('b', 0.4),
    'resnet152-group': 'b',
}

line_styles = {
    'alexnet-fixed': '--',
    'alexnet-group': '-',
    'vgg19_bn-fixed': '--',
    'vgg19_bn-group': '-',
    'shiftnet19-fixed': '--',
    'shiftnet19-group': '-',
    'resnet152-fixed': '--',
    'resnet152-group': '-',
}

marker_styles = {
    'alexnet-fixed': 'o',
    'alexnet-group': 'o',
    'vgg19_bn-fixed': '^',
    'vgg19_bn-group': '^',
    'shiftnet19-fixed': 'o',
    'shiftnet19-group': 'o',
    'resnet152-fixed': 'v',
    'resnet152-group': 'v',
}



plt.figure(figsize=(5, 4.5))
for name in names:
    with open('data/{}-results.txt'.format(name), 'r') as fp:
        model_results = json.load(fp)

    for key, value in model_results.items():
        label = '{} ({})'.format(fancy_names[name], fancy_names[key])
        color = colors['{}-{}'.format(name, key)]
        linestyle = line_styles['{}-{}'.format(name, key)]
        markerstyle = marker_styles['{}-{}'.format(name, key)]
        terms, accs = [], []
        for term, acc in zip(value['avg_terms'], value['acc']):
            if  term >= 1 and term <= 3:
                terms.append(term)
                accs.append(acc)
        plt.plot(terms, accs, '-o', linewidth=2, ms=8, color=color,
                 linestyle=linestyle, marker=markerstyle,
                 markeredgecolor='k', markeredgewidth=0.8, label=label)
    

plt.legend(loc=0)
plt.tight_layout()
plt.title('(b) Versus Term Truncation')
plt.xlabel(r'Average Number of Terms $\alpha$')
plt.savefig('figures/model-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()