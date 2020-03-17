import json

import util
plt = util.import_plt_settings(local_display=True)
import matplotlib
matplotlib.rc('legend', fontsize=12)

names = ['resnet18']
fancy_names = {'static': 'Static', 'dynamic': 'Dynamic', 'alexnet': 'AlexNet', 'vgg19_bn': 'VGG-19',
               'shiftnet19': 'ShiftNet-19', 'resnet152': 'ResNet-152', 'resnet18': 'ResNet-18'}

colors = {
    'resnet18-static': 'r',
    'resnet18-dynamic': 'b',
}

line_styles = {
    'resnet18-static': '--',
    'resnet18-dynamic': '-',
}

marker_styles = {
    'resnet18-static': 'o',
    'resnet18-dynamic': 'o',
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
        for term, acc in zip(value['avg_terms'], value['accs']):
            if  term >= 1 and term <= 3:
                terms.append(term)
                accs.append(acc)
        plt.plot(terms, accs, '-o', linewidth=2, ms=8, color=color,
                 linestyle=linestyle, marker=markerstyle,
                 markeredgecolor='k', markeredgewidth=0.8, label=label)
    

plt.legend(loc=0)
plt.tight_layout()
plt.title('Static vs Dynamic TR')
plt.xlabel(r'Average Number of Terms $\alpha$')
plt.savefig('figures/static-v-dynamic-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()