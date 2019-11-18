import numpy as np

import util
plt = util.import_plt_settings(local_display=True)

area_eff = {
    'bmac': [1, 1, 1],
    'pmac': [1.8, 1.8, 1.8],
    'tmac': [1.7, 2.4, 2.8],
}

energy_eff = {
    'bmac': [1, 1, 1],
    'pmac': [2, 2, 2],
    'tmac': [2.2, 3.0, 3.4],
}


x_ticks = np.array([0, 1, 2])
width = 0.25

plt.figure(figsize=(6, 4.5))
plt.bar(x_ticks, area_eff['bmac'], label='bMAC', width=width, edgecolor='k', color='tan')
plt.bar(x_ticks+width, area_eff['pmac'], label='pMAC', hatch='/', width=width, edgecolor='k', color='lightsteelblue')
plt.bar(x_ticks+2*width, area_eff['tmac'], label='tMAC', hatch='//', width=width, edgecolor='k', color='tomato')

plt.legend(loc=0)
plt.tight_layout()
plt.title('(a) ASIC Area Efficiency')
plt.ylabel('Normalized Area Efficiency')
plt.xticks(x_ticks+width, [r'$\alpha=2,g=2$', r'$\alpha=1.25,g=8$',r'$\alpha=1,g=32$'], fontsize=17)
plt.savefig('figures/asic-area-efficiency.pdf', dpi=300, bbox_inches='tight')
plt.clf()


plt.figure(figsize=(6, 4.5))
plt.bar(x_ticks, energy_eff['bmac'], label='bMAC', width=width, edgecolor='k', color='tan')
plt.bar(x_ticks+width, energy_eff['pmac'], label='pMAC', hatch='/', width=width, edgecolor='k', color='lightsteelblue')
plt.bar(x_ticks+2*width, energy_eff['tmac'], label='tMAC', hatch='//', width=width, edgecolor='k', color='tomato')

plt.legend(loc=0)
plt.tight_layout()
plt.title('(b) ASIC Energy Efficiency')
plt.ylabel('Normalized Energy Efficiency')
plt.xticks(x_ticks+width, [r'$\alpha=2,g=2$', r'$\alpha=1.25,g=8$',r'$\alpha=1,g=32$'], fontsize=17)
plt.savefig('figures/asic-energy-efficiency.pdf', dpi=300, bbox_inches='tight')
plt.clf()
