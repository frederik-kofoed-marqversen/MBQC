import json
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

import np_quantum as npq
import variational_forms as vform
import vehicle_routing

penalty_factor = 2.5
num_runs = 100
parallelise = True
DATA_DIR = '../data/tests/vehicle_routing/results.txt'
hamiltonian, vertices = vehicle_routing.get_hamiltonian(penalty_factor)
num_entangling_steps = 1
num_qubits = npq.num_qubits(hamiltonian)
ansatz_circuit = vform.get_vqe_circ(num_qubits, num_entangling_steps, vform.cnot_cyl_entangling)
num_parameters = vform.number_of_parameters(ansatz_circuit)


def write_data(string: str):
    with open(DATA_DIR, 'a') as file:
        file.write(string)


def get_variational_state(angles):
    return vform.mb_variational_form(ansatz_circuit, angles)


def get_variational_energy(angles):
    return npq.expecth(hamiltonian, get_variational_state(angles))


def minimise_energy(initial_guess):
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x},  # theta > 0
        {'type': 'ineq', 'fun': lambda x: 2 * np.pi - x}  # theta < 2*pi
    ]

    min_energy_angles = minimize(
        get_variational_energy,
        initial_guess,
        method='COBYLA',  # SLSQP, COBYLA
        # bounds=[(0, 2 * np.pi)] * len(initial_guess),  # only for SLSQP
        constraints=constraints,  # only for COBYLA
        options={
            'maxiter': 2000,
            'rhobeg': np.pi / 2
        }
    ).x

    # energy = get_variational_energy(min_energy_angles)
    state_probabilities = np.abs(vform.mb_variational_form(ansatz_circuit, min_energy_angles)) ** 2
    return state_probabilities


states = [format(i, f'0{str(num_qubits)}b') for i in range(2 ** num_qubits)]
probabilities = np.zeros(len(states))

remaining_processes = num_runs
print(f'Running MBVQE {num_runs} times:')
with open(DATA_DIR, 'w') as file:  # clears contents
    pass
if parallelise:
    initial_guesses = [np.random.rand(num_parameters) * 2 * np.pi for _ in range(num_runs)]
    pool = Pool(processes=min(3, num_runs))
    for result in pool.imap_unordered(minimise_energy, initial_guesses):
        write_data(json.dumps(result.tolist()) + '\n')
        probabilities += result
        remaining_processes -= 1
        print(f'    process closed ({remaining_processes} remaining)')
else:
    for i in range(num_runs):
        initial_guess = np.random.rand(num_parameters) * 2 * np.pi
        result = minimise_energy(initial_guess)
        write_data(json.dumps(result.tolist()) + '\n')
        probabilities += result
        remaining_processes -= 1
        print(f'    process closed ({remaining_processes} remaining)')
print()
probabilities /= num_runs

# sort propabilities
sorted_list = sorted(zip(states, probabilities), key=lambda x: x[1], reverse=True)

print(vertices)
print(
    [(vehicle_routing.get_graph_result(string, len(vertices) - 1), round(prob, 5)) for string, prob in sorted_list[:5]])
