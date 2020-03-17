import numpy as np

import util
plt = util.import_plt_settings(local_display=True)

plt.rc('axes', titlesize=25)  # fontsize of the figure title

energy_eff = {
    'bmac': [1, 1, 1],
    'pmac': [2.5, 2.5, 2.5],
    'tmac': [2.7, 3.3, 3.6],
}


x_ticks = np.array([0, 1, 2])

width = 0.25
plt.figure(figsize=(6.5, 4.5))
plt.bar(x_ticks, energy_eff['bmac'], label='bMAC', width=width, edgecolor='k', color='tan')
plt.bar(x_ticks+width, energy_eff['pmac'], label='pMAC', hatch='/', width=width, edgecolor='k', color='lightsteelblue')
plt.bar(x_ticks+2*width, energy_eff['tmac'], label='tMAC', hatch='//', width=width, edgecolor='k', color='tomato')

plt.legend(loc=0)
plt.tight_layout()
plt.title('(d) FPGA Energy Efficiency')
# plt.ylim(top=100)
plt.ylabel('Normalized Energy Efficiency')
plt.xticks(x_ticks+width, [r'$\alpha=2,g=2$', r'$\alpha=1.25,g=8$',r'$\alpha=1,g=32$'], fontsize=17)
plt.savefig('figures/fpga-power.pdf', dpi=300, bbox_inches='tight')
plt.clf()