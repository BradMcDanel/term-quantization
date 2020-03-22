import json

import util
plt = util.import_plt_settings(local_display=True)
import matplotlib
matplotlib.rc('legend', fontsize=12)

names = ['resnet18', 'vgg16_bn', 'mobilenet_v2', 'efficientnet_b0']
# names = ['mobilenet_v2']
fancy_names = {'static': 'Static', 'dynamic': 'Dynamic', 'alexnet': 'AlexNet', 'vgg19_bn': 'VGG-19',
               'shiftnet19': 'ShiftNet-19', 'resnet152': 'ResNet-152', 'resnet18': 'ResNet-18',
               'quant': 'Quant', 'tr-data2': 'TR-2', 'tr-data3': 'TR-3', 'vgg16_bn': 'VGG-16',
               'mobilenet_v2': 'MobileNetV2', 'efficientnet_b0': 'EfficientNet-b0'}

colors = {
    'resnet18-quant': 'r',
    'resnet18-tr-data2': 'r',
    'resnet18-tr-data3': 'r',
    'mobilenet_v2-quant': 'b',
    'mobilenet_v2-tr-data2': 'b',
    'mobilenet_v2-tr-data3': 'b',
    'mobilenet_v2-tr-data4': 'b',
    'efficientnet_b0-quant': 'k',
    'efficientnet_b0-tr-data2': 'k',
    'efficientnet_b0-tr-data3': 'k',
    'efficientnet_b0-tr-data4': 'k',
    'vgg16_bn-quant': 'g',
    'vgg16_bn-tr-data2': 'g',
    'vgg16_bn-tr-data3': 'g',
}

line_styles = {
    'resnet18-quant': '-',
    'resnet18-tr-data2': '',
    'resnet18-tr-data3': '',
    'mobilenet_v2-quant': '-',
    'mobilenet_v2-tr-data2': '',
    'mobilenet_v2-tr-data3': '',
    'mobilenet_v2-tr-data4': '',
    'efficientnet_b0-quant': '-',
    'efficientnet_b0-tr-data2': '',
    'efficientnet_b0-tr-data3': '',
    'efficientnet_b0-tr-data4': '',
    'vgg16_bn-quant': '-',
    'vgg16_bn-tr-data2': '',
    'vgg16_bn-tr-data3': '',
}

marker_styles = {
    'resnet18-quant': 'o',
    'resnet18-tr-data2': '^',
    'resnet18-tr-data3': '^',
    'mobilenet_v2-quant': 'o',
    'mobilenet_v2-tr-data2': '^',
    'mobilenet_v2-tr-data3': '^',
    'mobilenet_v2-tr-data4': '^',
    'efficientnet_b0-quant': 'o',
    'efficientnet_b0-tr-data2': '^',
    'efficientnet_b0-tr-data3': '^',
    'efficientnet_b0-tr-data4': '^',
    'vgg16_bn-quant': 'o',
    'vgg16_bn-tr-data2': '^',
    'vgg16_bn-tr-data3': '^',
}

plt.figure(figsize=(8, 6))
for name in names:
    with open('data/{}-results.txt'.format(name), 'r') as fp:
        model_results = json.load(fp)

    for key, value in model_results.items():
        if 'data' in key and '2' in key:
            continue
        elif 'data' in key and '3' in key:
            label = '{} (TR)'.format(fancy_names[name])
        elif 'data' in key:
            label = ''
        else:
            label = '{} ({})'.format(fancy_names[name], fancy_names[key])
        color = colors['{}-{}'.format(name, key)]
        linestyle = line_styles['{}-{}'.format(name, key)]
        markerstyle = marker_styles['{}-{}'.format(name, key)]
        terms, accs = [], []
        # plt.plot(value['tmacs'], value['accs'], '-o', linewidth=2, ms=4, color=color,
        #          linestyle=linestyle, marker=markerstyle,
        #          markeredgecolor='k', markeredgewidth=0.8, label=label)
        print(key)
        plt.plot(value['tmacs'], value['accs'], linewidth=2, ms=6, color=color,
                 linestyle=linestyle, marker=markerstyle,
                 markeredgecolor='k', markeredgewidth=0.8, label=label)
    

plt.legend(loc=0)
plt.tight_layout()
plt.title('Quantization vs Term Revealing')
plt.xlabel(r'Exponent Additions per Sample')
plt.xscale('log')
plt.ylim(67,78)
plt.savefig('figures/static-v-dynamic-accuracy.pdf', dpi=300, bbox_inches='tight')
plt.clf()