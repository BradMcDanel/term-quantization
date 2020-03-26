import json

import matplotlib

import visualize

matplotlib.rc('legend', fontsize=12)
plt = visualize.import_plt_settings(local_display=True)

names = ['mnist-quant', 'mnist-tr']
fancy_names = {'mnist': 'MLP (MNIST)', 'quant': 'Quant', 'tr': 'TR'}

colors = {
    'mnist-quant': 'b',
    'mnist-tr': 'g',
}

line_styles = {
    'mnist-quant': '-',
    'mnist-tr': '-',
}

marker_styles = {
    'mnist-quant': 'o',
    'mnist-tr': 'o',
}

plt.figure(figsize=(8, 6))
for name in names:
    model, setting = name.split('-')
    with open('results/{}.json'.format(name), 'r') as fp:
        value = json.load(fp)

    label = '{} ({})'.format(fancy_names[model], fancy_names[setting])
    color = colors[name]
    linestyle = line_styles[name]
    markerstyle = marker_styles[name]
    plt.plot(value['tmacs'], value['accs'], linewidth=2, ms=6, color=color,
             linestyle=linestyle, marker=markerstyle,
             markeredgecolor='k', markeredgewidth=0.8, label=label)


plt.legend(loc=0)
plt.tight_layout()
plt.title('Quantization vs Term Revealing')
plt.xlabel('Exponent Additions per Sample')
plt.xscale('log')
# plt.ylim(67, 78)
plt.savefig('figures/mlp-comparison.pdf', dpi=300, bbox_inches='tight')
plt.clf()
