import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from spin_systems import spin_lattice_hamiltonian

n, m = 2, 2


def hamiltonian(J):
    return spin_lattice_hamiltonian(n, m, J, 1)


def number_of_similar_entries(lst):
    distinct_entries = sorted(set(lst))
    return distinct_entries, [sum(entry == test for entry in lst) for test in distinct_entries]


# CUSTOM COLORMAP
colours = np.linspace(1, 0, 256)
colours = colours ** 1.1
colours = np.tile(colours, (3, 1)).transpose()
colour_array = np.ones((256, 4))
colour_array[:, :-1] = colours
my_colourmap = ListedColormap(colour_array)

# PLOTTING
path = '../data/spin_2cx.txt'
title = r'MBVQE (4 CX)'
fig_title = 'Depth 2 MBVQE'
plt.rcParams.update({
    'text.usetex': True,
    'font.size': 12
})

fig, ax = plt.subplots(1, 1, figsize=[6, 4])

histograms = []
data = np.array(json.loads(open(path, 'r').read().splitlines()[1]))
data_x = data[:, 0]
data_y = data[:, 1]
xs = set(data_x)
num_xs = len(xs)
num_ys = 50
num_points_per_x = len(data_x) / num_xs
x_dist = (max(xs) - min(xs)) / num_xs
x_range = [min(xs) - x_dist / 2, max(xs) + x_dist / 2]
y_dist = (max(data_y) - min(data_y)) / num_ys
y_range = [min(data_y) - y_dist / 2, max(data_y) + y_dist]
_, _, _, h = ax.hist2d(data_x, data_y, range=[x_range, y_range], bins=[num_xs, num_ys],
                       weights=np.ones(len(data_x)) / num_points_per_x, cmap=my_colourmap, vmin=0, vmax=1)
histograms.append(h)

print(title + ':')
print(f'    avg number of points per x: {round(len(data_y) / len(xs))}')

# EXACT DIAGONALISATION
end_x = 1.1
xs = np.linspace(0, end_x, 150)
ys = np.array([np.sort(np.linalg.eigvalsh(hamiltonian(J)))[0] for J in xs])
ax.plot(xs, ys, 'k-', label='Exact', zorder=3, linewidth=1.5)

# SEPARABLE GROUND STATE
end_x = 1.1
filepath = '../data/spin_separable.txt'
data = np.array(json.loads(open(filepath, 'r').read().splitlines()[1]))
data_x = data[:, 0]
data_y = data[:, 1]
if data_x[-1] < end_x:  # add a bit extra so line hits the right edge
    data_y = np.append(data_y,
                       data_y[-1] + (end_x - data_x[-1]) * (data_y[-1] - data_y[-2]) / (data_x[-1] - data_x[-2]))
    data_x = np.append(data_x, end_x)
ax.plot(data_x, data_y, linestyle=(0, (3, 5)), color='0', label='Separable', zorder=2, linewidth=1.5)

# FINIALISE FIGURE

ax.legend(loc='lower left', borderpad=1.2, title=title)
ax.set_ylabel('Energy (a.u.)')
ax.set_xlabel('Spin coupling strength (J/B)')
ax.set_ylim(-7.9, -2.05)
fig.suptitle(fig_title)
plt.tight_layout()
cbar = fig.colorbar(histograms[-1], ax=ax, cmap=my_colourmap, aspect=35)
# cbar.set_label('Outcome probability')

plt.show()
