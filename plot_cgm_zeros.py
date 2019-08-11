import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

SMALL_SIZE = 13
MEDIUM_SIZE = 17
BIGGER_SIZE = 20
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the figure title

with open('alexnet.pkl', 'rb') as f:
    alexnet_zeros = pickle.load(f)

with open('alexnet_cgm2.pkl', 'rb') as f:
    alexnetcgm_zeros = pickle.load(f)


for i in range(len(alexnet_zeros)):
    d = np.array(alexnet_zeros[i])
    plt.plot(100.*(d[::2] + d[1:][::2]), linewidth=2.5, label='AlexNet+ReLU')
    d = np.array(alexnetcgm_zeros[i])
    plt.plot(100.*(d[::2] + d[1:][::2]), linewidth=2.5, label='AlexNet+CGM')
    plt.legend(loc=0)
    plt.title('AlexNet: Layer {}'.format(i+1))
    plt.xlabel('Activation Channel Pair Index')
    plt.ylabel('Sparsity (%) for Pairs of Channels')
    plt.savefig('figures/relu-vs-cgm-pairs-{}.png'.format(i+1), dpi=300)
    plt.clf()

for i in range(len(alexnet_zeros)):
    plt.plot(100.*np.array(sorted(alexnet_zeros[i])), linewidth=2.5, label='AlexNet+ReLU')
    plt.plot(100.*np.array(sorted(alexnetcgm_zeros[i])), linewidth=2.5, label='AlexNet+CGM')
    plt.legend(loc=0)
    plt.title('AlexNet: Layer {}'.format(i+1))
    plt.xlabel('Activation Channel Index (sorted)')
    plt.ylabel('Sparsity (%)')
    plt.savefig('figures/relu-vs-cgm-sparsity-{}.png'.format(i+1), dpi=300)
    plt.clf()