import json

import matplotlib.pyplot as plt
import numpy as np

eigenenergies = json.loads(open('../data/h2_eigenenergies.txt', 'r').read())
eig_xs = np.array(eigenenergies.pop('distances'))
hartree_fock_energies = eigenenergies.pop('hartree_fock')

# PLOTTING
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 11
})

# fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=[6.4 * 2, 3.8])
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=[6.4, 3 * 2])

filepath = '../data/h2_kink.txt'
data = json.loads(open(filepath, 'r').read().splitlines()[-1])
distances = json.loads(open(filepath, 'r').read().splitlines()[1])
axs[0].plot(distances, data, 'o', color='0', markersize=7, fillstyle='none', markeredgewidth=2,
            label=r'MBVQE (4 CX)', zorder=3)

filepath = '../data/h2_zn.txt'
data = json.loads(open(filepath, 'r').read().splitlines()[-1])
distances = json.loads(open(filepath, 'r').read().splitlines()[1])
axs[1].plot(distances, data, '*', color='0', markersize=10, fillstyle='full', markeredgewidth=2,
            label=r'MBVQE ($R_{Z^{\otimes 4}}$)', zorder=3)

for ax in axs:
    ax.plot(eig_xs, eigenenergies['singlet'], 'k-', label=r'Exact', zorder=2)
    ax.plot(eig_xs, eigenenergies['triplet'], 'k-', zorder=2)
    ax.plot(eig_xs, hartree_fock_energies, linestyle='--', color='0.6', label=r'Hartree-Fock', zorder=1)
    ax.legend(loc='lower right', borderpad=1.2)

axs[0].set_ylim(-1.18, -0.72)
axs[0].set_xlim(0.25, 2.6)
#axs[0].set_xlabel('Interatomic distance (Å)')
axs[1].set_xlabel('Interatomic distance (Å)')
axs[0].set_ylabel('Energy (Hartree)')
axs[1].set_ylabel('Energy (Hartree)')

plt.tight_layout()
plt.show()
