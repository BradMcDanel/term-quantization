import json

import matplotlib

import visualize

matplotlib.rc('legend', fontsize=12)
plt = visualize.import_plt_settings(local_display=True)

def gen_frontier(xs, ys):
    xs, ys = zip(*sorted(zip(xs, ys)))
    curr_y = float('-inf')
    ret_xs, ret_ys = [], []
    for x, y in zip(xs, ys):
        if y > curr_y:
            ret_xs.append(x)
            ret_ys.append(y)
            curr_y = y

    return ret_xs, ret_ys


cnn_names = ['resnet18', 'vgg16_bn', 'mobilenet_v2', 'efficientnet_b0']
fancy_names = {'static': 'Static', 'dynamic': 'Dynamic', 'alexnet': 'AlexNet', 'vgg19_bn': 'VGG-19',
               'shiftnet19': 'ShiftNet-19', 'resnet152': 'ResNet-152', 'resnet18': 'ResNet-18',
               'quant': 'Quant', 'vgg16_bn': 'VGG-16',
               'mobilenet_v2': 'MoblNet-v2', 'efficientnet_b0': 'EffNet-b0'}

colors = {
    'mnist-quant': 'k',
    'mnist-tr': 'k',
    'lstm-quant': 'purple',
    'lstm-tr': 'purple',
    'resnet18-quant': 'r',
    'resnet18-tr': 'r',
    'mobilenet_v2-quant': 'b',
    'mobilenet_v2-tr': 'b',
    'efficientnet_b0-quant': 'orange',
    'efficientnet_b0-tr': 'orange',
    'vgg16_bn-quant': 'g',
    'vgg16_bn-tr': 'g',
}

line_styles = {
    'mnist-quant': '-',
    'mnist-tr': '-.',
    'lstm-quant': '-',
    'lstm-tr': '-.',
    'resnet18-quant': '-',
    'resnet18-tr': '-.',
    'mobilenet_v2-quant': '-',
    'mobilenet_v2-tr': '-.',
    'efficientnet_b0-quant': '-',
    'efficientnet_b0-tr': '-.',
    'vgg16_bn-quant': '-',
    'vgg16_bn-tr': '-.',
}

marker_styles = {
    'mnist-quant': 'o',
    'mnist-tr': '^',
    'lstm-quant': 'o',
    'lstm-tr': '^',
    'resnet18-quant': 'o',
    'resnet18-tr': '^',
    'mobilenet_v2-quant': 'o',
    'mobilenet_v2-tr': '^',
    'efficientnet_b0-quant': 'o',
    'efficientnet_b0-tr': '^',
    'vgg16_bn-quant': 'o',
    'vgg16_bn-tr': '^',
}

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5),
                                  gridspec_kw={
                                      'width_ratios': [2, 4, 2],
                                  })

# MLP
with open('results/mnist-quant.json', 'r') as fp:
    quant_results = json.load(fp)

with open('results/mnist-tr.json', 'r') as fp:
    tr_results = json.load(fp)

quant_key = 'mnist-quant'
tr_key = 'mnist-tr'

# plot quant setting
ax1.plot(quant_results['tmacs'], quant_results['accs'], linewidth=2, ms=6,
         color=colors[quant_key], linestyle=line_styles[quant_key],
         marker=marker_styles[quant_key], markeredgecolor='k',
         markeredgewidth=0.8, label='MLP (QT)')

# plot tr TR
ax1.plot(tr_results['tmacs'], tr_results['accs'], linewidth=2, ms=6,
         color=colors[tr_key], linestyle=line_styles[tr_key],
         marker=marker_styles[tr_key], markeredgecolor='k',
         markeredgewidth=0.8, label='MLP (TR)')

ax1.set_title('MLP (MNIST)')
ax1.set_xscale('log')
ax1.set_ylabel('Top-1 Accuracy (%)')
ax1.set_xticks([10**6, 10**7])


# CNNs
for name in cnn_names:
    with open('results/{}-results.json'.format(name), 'r') as fp:
        model_results = json.load(fp)

    # keys used for plotting settings
    quant_key = '{}-{}'.format(name, 'quant')
    tr_key = '{}-{}'.format(name, 'tr')

    # quant settings
    quant_accs = model_results['quant']['accs']
    quant_tmacs = model_results['quant']['tmacs']

    # aggregate tr settings
    tr_accs, tr_tmacs = [], []
    for key, value in model_results.items():
        if 'data' in key:
            tr_accs.extend(value['accs'])
            tr_tmacs.extend(value['tmacs'])

    tr_tmacs, tr_accs = gen_frontier(tr_tmacs, tr_accs)


    # plot quant setting
    ax2.plot(quant_tmacs, quant_accs, linewidth=2, ms=6,
             color=colors[quant_key], linestyle=line_styles[quant_key],
             marker=marker_styles[quant_key], markeredgecolor='k',
             markeredgewidth=0.8, label=fancy_names[name] + ' (QT)')

    # plot tr TR
    ax2.plot(tr_tmacs, tr_accs, linewidth=2, ms=6,
             color=colors[tr_key], linestyle=line_styles[tr_key],
             marker=marker_styles[tr_key], markeredgecolor='k',
             markeredgewidth=0.8, label=fancy_names[name] + ' (TR)')

ax2.set_title('CNNs (ImageNet)')
# ax2.set_ylabel('Top-1 Accuracy (%)')
ax2.set_xscale('log')
ax2.set_ylim(67, 77)
ax2.set_xlabel('Term Pair Multiplications per Sample (log scale)')

# LSTM
with open('results/lstm-quant.json', 'r') as fp:
    quant_results = json.load(fp)

with open('results/lstm-tr.json', 'r') as fp:
    tr_results = json.load(fp)

quant_key = 'lstm-quant'
tr_key = 'lstm-tr'

# plot quant setting
ax3.plot(quant_results['tmacs'], quant_results['ppls'], linewidth=2, ms=6,
         color=colors[quant_key], linestyle=line_styles[quant_key],
         marker=marker_styles[quant_key], markeredgecolor='k',
         markeredgewidth=0.8, label='LSTM (QT)')

# plot tr TR
ax3.plot(tr_results['tmacs'], tr_results['ppls'], linewidth=2, ms=6,
         color=colors[tr_key], linestyle=line_styles[tr_key],
         marker=marker_styles[tr_key], markeredgecolor='k',
         markeredgewidth=0.8, label='LSTM (TR)')

ax3.set_xscale('log')
ax3.set_ylabel('Perplexity')
ax3.set_xlim(xmax=10**12)
ax3.set_title('LSTM (Wiki-2)')

f.legend(loc='center left', frameon=False)
f.tight_layout(pad=0)
plt.subplots_adjust(left=0.28, wspace=0.65)

pos1 = ax1.get_position()
pos2 = ax2.get_position()

points2 = pos2.get_points()
points2[0][0] -= .055
pos2.set_points(points2)
ax2.set_position(pos2)

plt.savefig('figures/tr-comp.pdf', dpi=300, bbox_inches='tight')
plt.clf()
