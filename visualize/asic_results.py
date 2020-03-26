import numpy as np

import util
plt = util.import_plt_settings(local_display=True)

plt.rc('axes', titlesize=25)  # fontsize of the figure title

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


ees = [2.88, 3.50, 2.82, 1.70]
lats = [1.34, 1.87, 1.24, 1.10]
xs = np.arange(len(ees))
fig = plt.figure(figsize=(6, 2.5))
ax = fig.add_subplot(111)
ax2 = ax.twinx()
width = 0.3

ax.bar(xs, ees, color='lightblue', edgecolor='k', width=width, label='Energy Efficieny')
ax2.bar(xs + width, lats, color='cornflowerblue', edgecolor='k', hatch='/', width=width, label='Latency')

ax.set_ylabel('Energy Efficiency')
ax.set_ylim(1, 4)
ax2.set_ylabel('Latency')
ax2.set_ylim(1, 2)

plt.xticks(xs + width/2, ['AlexNet', 'VGG-16', 'GoogleNet', 'MobileNet-V2'])
ax.tick_params(axis='x', labelsize=12)

          
fig.legend(bbox_to_anchor=(0.88, 0.92), loc='upper right', ncol=1)

plt.savefig('figures/asic-laconic.pdf', dpi=300, bbox_inches='tight')
plt.clf()
