import json
import random
import time
from typing import Callable

import tqdm
from scipy.linalg import expm

import chemistry_package as chem
import mbqgates
from graph_class_v2 import Graph
from graph_state_tn import GraphStateTensorNetwork
from mbqc_v2 import MBQCircuit, simulate_circuit
from np_quantum import *


class Colour:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def colour_this(cls, colour_this, colour_identifier: str = None):
        if colour_identifier is None:
            return str(colour_this)
        else:
            return getattr(cls, colour_identifier) + str(colour_this) + cls.ENDC

    @classmethod
    def bool_colour(cls, bool: bool, colour_this=None):
        if bool:
            colour = cls.OKGREEN
        else:
            colour = cls.FAIL
        if colour_this is None:
            string = str(bool)
        else:
            string = str(colour_this)
        return cls.BOLD + colour + string + cls.ENDC


def tensor_network_cnot():
    input_states = [np.random.rand(2) for _ in range(2)]

    network = GraphStateTensorNetwork(Graph.grid(3, 7))
    for col in [0, 1, 2, 4, 5, 6]:
        network.add_pauli_measurement((1, col), 'Z', 0)
    for qubit in [(0, i) for i in range(1, 6)] + [(1, 3), (2, 3)]:
        network.add_pauli_measurement(qubit, 'Y', 0)
    for col in [1, 2, 4, 5]:
        network.add_pauli_measurement((2, col), 'X', 0)

    for i, qubit in enumerate([(0, 0), (2, 0)]):  # input qubits
        network.add_pauli_measurement(qubit, 'X', 0)
        network.set_initial_state(qubit, input_states[i])

    result = network.get_state()
    result = np.kron(Z, IDTY) @ result
    result = ket2dm(result)

    goal = CNOT @ np.kron(*input_states)
    goal = ket2dm(normalise(goal))

    return np.allclose(result, goal)


def one_way_computer():
    circ = MBQCircuit(2)
    initial_state = [rand_ket() for _ in range(2)]

    cnot = mbqgates.cnot()
    circ.add_gate(cnot, [0, 1])
    circ.add_gate(cnot, [0, 1])

    simulated_result = simulate_circuit(circ, initial_state)
    goal = tensor(*initial_state)

    return compare_kets(simulated_result, goal)


def vqe_entangling():
    input_states = [rand_ket() for _ in range(3)]

    circ = MBQCircuit(3)
    cnot = mbqgates.cnot()
    circ.add_gate(cnot, [0, 1])
    circ.add_gate(cnot, [0, 2])
    circ.add_gate(cnot, [1, 2])

    simulated_result = simulate_circuit(circ, input_states)
    goal = expand_gate(CNOT, 3, [1, 2]) @ expand_gate(CNOT, 3, [0, 2]) @ expand_gate(CNOT, 3, [0, 1]) @ tensor(
        *input_states)

    return compare_kets(simulated_result, goal)


def rotation():
    num_lines = 2
    angles = np.random.rand(num_lines, 3)
    initial_states = [rand_ket() for _ in range(num_lines)]

    circ = MBQCircuit(num_lines)
    gate = tensor(*[IDTY] * num_lines)
    for i in range(num_lines):
        circ.add_gate(mbqgates.euler_rotation(*angles[i]), [i])
        gate = expand_gate(euler_rotation(*angles[i]), num_lines, [i]) @ gate

    simulated_result = simulate_circuit(circ, initial_states)
    expected_result = gate @ tensor(*initial_states)

    return compare_kets(expected_result, simulated_result)


def vqe_layer():
    # at 4 lines: 40 it/s with efficient pauli simulation, and 20 it/s without
    num_lines = 4
    angles = np.random.rand(num_lines, 3)
    initial_states = [rand_ket(2) for _ in range(num_lines)]

    circ = MBQCircuit(num_lines)
    gate = tensor(*[IDTY] * num_lines)
    for i in range(num_lines - 1):
        circ.add_gate(mbqgates.cnot(), [i, i + 1])
        gate = expand_gate(CNOT, num_lines, [i, i + 1]) @ gate
    for i in range(num_lines):
        circ.add_gate(mbqgates.euler_rotation(*angles[i]), [i])
        gate = expand_gate(euler_rotation(*angles[i]), num_lines, [i]) @ gate

    circ.remove_pauli_measurements()
    simulated_result = simulate_circuit(circ, initial_states)
    expected_result = gate @ tensor(*initial_states)

    return compare_kets(expected_result, simulated_result)


def entangler_gate():
    num_lines = 4
    initial_states = [rand_ket() for _ in range(num_lines)]

    circ = MBQCircuit(num_lines)
    circ.add_gate(mbqgates.mb_entangler(num_lines), list(range(num_lines)))

    result = True
    simulated_result = simulate_circuit(circ, initial_states)
    for i in range(5):
        simulated_result_2 = simulate_circuit(circ, initial_states)
        if not compare_kets(simulated_result_2, simulated_result):
            # gate has produced non deterministic result
            result = False
            break

    return result


def multi_qubit_phase_gate():
    num_lines = 4
    initial_states = [rand_ket() for _ in range(num_lines)]
    angle = np.random.rand() * 2 * np.pi

    apply_to = list(range(num_lines))
    not_apply_to = random.choice(apply_to)
    apply_to.remove(not_apply_to)

    circ = MBQCircuit(num_lines)
    circ.add_gate(mbqgates.cn_phase(angle, len(apply_to)), apply_to)

    gate = tensor(*[Z] * (num_lines - 1))
    gate = expm(-1j * angle / 2 * gate)
    print(gate)
    exit()
    gate = expand_gate(gate, num_lines, apply_to)

    simulated_result = simulate_circuit(circ, initial_states)
    expected_result = gate @ tensor(*initial_states)

    return compare_kets(simulated_result, expected_result)


def compute_h2_eigenenergies():
    interval = (0.2, 4)
    num_points = 100
    distances = np.linspace(*interval, num_points)
    result = np.array([chem.h2_sorted_energies(dist) + [chem.h2_hartree_fock_energy(dist)] for dist in distances])
    dictionary = dict(singlet=result[:, 0].tolist(), triplet=result[:, 1].tolist(), ion=result[:, 2].tolist(),
                      hartree_fock=result[:, 3].tolist(), distances=distances.tolist())
    print(json.dumps(dictionary))


def main():
    def my_n_times(function, nums):
        successes = 0
        time.sleep(0.1)
        for _ in tqdm.trange(nums):
            if function():
                successes += 1
        return successes

    def my_runs(*functions: Callable, **kwargs):
        eval_num = 1
        if 'eval_num' in kwargs:
            eval_num = kwargs['eval_num']

        def print_spacing():
            print()
            print('─' * 50)
            print()

        print_spacing()
        for function in functions:
            if eval_num == 1:
                print(Colour.colour_this(f'NOW RUNNING {function.__name__}()', 'BOLD'))
                print()
                result = function()
                print()
                print(f'Success: {Colour.bool_colour(result)}')
            else:
                print(Colour.colour_this(f'NOW RUNNING {function.__name__}() {eval_num} TIMES', 'BOLD'))
                print()
                num_successes = my_n_times(function, eval_num)
                print()
                print(f'test failed {eval_num - num_successes} out of {eval_num} times')
                print(f'Success: {Colour.bool_colour(num_successes == eval_num)}')
            print_spacing()

    my_runs(one_way_computer, eval_num=100)
    compute_h2_eigenenergies()


if __name__ == '__main__':
    main()
