import numpy as np

import np_quantum as npq
from graph_class_v2 import Graph
from np_quantum import X, Y, Z


def spin_lattice_hamiltonian(n: int, m: int, J: float, B: float) -> np.ndarray:
    graph = Graph.grid(n, m)

    N = len(graph)
    qubit_state_ordering = list(graph.keys())
    hamiltonian = np.zeros((2 ** N, 2 ** N), dtype='complex128')

    hamiltonian += -B * sum(npq.expand_gate(Z, N, [i]) for i in range(N))
    operator = J * (npq.tensor(X, X) + npq.tensor(Y, Y) + npq.tensor(Z, Z))
    for qubit_pair in graph.get_edges():
        qubit_pair_index = [qubit_state_ordering.index(qubit) for qubit in qubit_pair]
        hamiltonian += npq.expand_gate(operator, N, qubit_pair_index)

    return hamiltonian


def spin_lattice_hamiltonian_true(n: int, m: int, J: float, B: float) -> np.ndarray:
    graph = Graph.grid(n, m)#.get_complete()

    N = len(graph)
    qubit_state_ordering = list(graph.keys())
    hamiltonian = np.zeros((2 ** N, 2 ** N), dtype='complex128')

    hamiltonian += -B * sum(npq.expand_gate(Z, N, [i]) for i in range(N))
    S = (X, Y, Z)
    for qubit_pair in graph.get_edges():
        rx, ry = (qubit_pair[0][i] - qubit_pair[1][i] for i in range(2))
        rz = 0
        r_vec = np.array([rx, ry, rz])
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r
        operator = sum(S[i] * r_hat[i] for i in range(3))
        ss_interaction = -J / (r ** 3) * (3 * npq.tensor(operator, operator) - sum(npq.tensor(S[i], S[i]) for i in range(3)))
        qubit_pair_index = [qubit_state_ordering.index(qubit) for qubit in qubit_pair]
        hamiltonian += npq.expand_gate(ss_interaction, N, qubit_pair_index)

    return hamiltonian


def total_spin_operator_squared(N):
    jx = sum(npq.expand_gate(X / 2, N, [i]) for i in range(N))
    jy = sum(npq.expand_gate(Y / 2, N, [i]) for i in range(N))
    jz = sum(npq.expand_gate(Z / 2, N, [i]) for i in range(N))
    j_vec = [jx, jy, jz]
    j_squared = sum(np.linalg.matrix_power(oper, 2) for oper in j_vec)
    return j_squared


def total_spin_z_operator(N):
    return sum(npq.expand_gate(Z / 2, N, [i]) for i in range(N))
