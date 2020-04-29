import numpy as np

import visualize
plt = visualize.import_plt_settings(local_display=True)

plt.rc('axes', titlesize=25)  # fontsize of the figure title

networks = ['MLP', 'VGG-16', 'ResNet-18', 'MoblNet-v2', 'EffNet-b0', 'LSTM']
latency = [6.2, 10.8, 8.8, 7.3, 8.1, 3.3]
energy_eff = [4.1, 7.0, 5.9, 4.6, 5.2, 2.1]

x_ticks = np.arange(len(networks))

width = 0.40
plt.figure(figsize=(9.5, 3.5))
plt.bar(x_ticks, energy_eff, label='Energy Efficency', width=width, edgecolor='k', color='lightblue')
plt.bar(x_ticks+width, latency, label='Latency', width=width, edgecolor='k', color='plum')

plt.legend(loc=0)
plt.tight_layout()
plt.title('FPGA improvements of TR over QT')
plt.ylabel('Normalized Improvement')
plt.xticks(x_ticks+(width/2), networks)
plt.savefig('figures/fpga-models.pdf', dpi=300, bbox_inches='tight')
plt.clf()
