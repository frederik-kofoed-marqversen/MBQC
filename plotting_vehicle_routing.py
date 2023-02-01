import json

import matplotlib.pyplot as plt
import numpy as np

import vehicle_routing as vehi

number_of_bars = 5
filepath = '../data/tests/vehicle_routing/vehicle_routing_1zn.txt'

data = open(filepath, 'r').read().splitlines()
data = [np.array(json.loads(string)) for string in data]

print(f'There are {len(data)} individual simulations.')
print('Plotting the average')
print()

avg_probabilities = sum(data) / len(data)
num_vertices = 3
num_qubits = num_vertices ** 2
states = [format(i, f'0{str(num_qubits)}b') for i in range(2 ** num_qubits)]


def reverse_state(state: str):
    return ''.join([state[i:i + num_vertices] for i in range(0, len(state), num_vertices)][::-1])


filtered_states = []
filtered_probabilities = []
for prob, state in zip(avg_probabilities, states):
    reversed_state = reverse_state(state)
    if reversed_state in filtered_states:
        index = filtered_states.index(reversed_state)
        filtered_probabilities[index] += prob
    else:
        filtered_states.append(state)
        filtered_probabilities.append(prob)

data_to_plot = sorted(zip(filtered_probabilities, filtered_states), key=lambda x: x[0], reverse=True)[0:number_of_bars]
# data_to_plot = sorted(zip(avg_probabilities, states), key=lambda x: x[0], reverse=True)[0:number_of_bars]
print(f'Only plotting the top {number_of_bars} probable states:')
print(data_to_plot)
print()

#  calculate standard deviation of plotted data
std_to_plot = []
for _, state in data_to_plot:
    probabilities = []
    index = states.index(state)
    for nparr in data:
        probabilities.append(nparr[index])
    reversed_state = reverse_state(state)
    index = states.index(reversed_state)
    for nparr in data:
        probabilities.append(nparr[index])
    std_to_plot.append(np.std(probabilities))

# PLOTTING

plt.rcParams.update({
    'text.usetex': True,
    'font.size': 12,
})
plt.figure(figsize=[6.4, 2.8])
# compute posisions of bars and x axis limits
bar_width = 1
dist_between_bars = 0.5
x_pos = [i * (bar_width + dist_between_bars) for i in range(len(data_to_plot))]
x_lim = (x_pos[0] - (dist_between_bars + bar_width / 2), x_pos[-1] + (dist_between_bars + bar_width / 2))
# plot
bar_heights = [x[0] for x in data_to_plot]
labels = [str(vehi.get_graph_result(state)) for _, state in data_to_plot]

print('The data:')
print([state for _, state in data_to_plot])
print(labels)
print([str(vehi.get_graph_result(reverse_state(state))) for _, state in data_to_plot])
print(bar_heights)
print([vehi.get_cost(state) for _, state in data_to_plot])
print([vehi.get_energy(state, 2.5) for _, state in data_to_plot])

plt.bar(x_pos, bar_heights, width=bar_width, color='0.6', tick_label=[''] * len(x_pos))
plt.errorbar(x_pos, bar_heights, yerr=std_to_plot, ecolor='black', capsize=4, fmt='.k')
# esthetics
plt.xlim(*x_lim)
plt.ylim(0, 0.75)
plt.ylabel('Outcome probability')
plt.tight_layout()
plt.show()
