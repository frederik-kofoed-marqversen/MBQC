import numpy as np

import np_quantum as npq
from graph_class_v2 import Graph


def get_weight(v1, v2):
    # euclidean_distance
    return (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2


def get_hamiltonian(penalty_factor: float = 0):
    start_vertex = (0, 0)
    graph = Graph.grid(2, 2).delete_vertex(start_vertex).get_complete()
    vertices_to_visit = list(graph.keys())
    n = len(vertices_to_visit)
    num_qubits = len(vertices_to_visit) ** 2

    def get_index(vertex, time):
        return vertices_to_visit.index(vertex) + time * n

    hamiltonian = np.zeros((2 ** num_qubits, 2 ** num_qubits))
    x = (npq.IDTY - npq.Z) / 2
    xx = npq.tensor(x, x)
    for v1 in graph:  # add cost of edges
        for v2 in graph:
            weight = get_weight(v1, v2)
            for t in range(n - 1):
                index1 = get_index(v1, t)
                index2 = get_index(v2, t + 1)
                hamiltonian += npq.expand_gate(weight * xx, num_qubits, [index1, index2])
    for v in vertices_to_visit:  # add penalty: vertex included for exactly one t
        oper = npq.expand_gate(npq.IDTY, num_qubits, [0])
        for t in range(n):
            oper += npq.expand_gate(-1 * x, num_qubits, [get_index(v, t)])
        hamiltonian += penalty_factor * (oper @ oper)
    for t in range(n):  # add penalty: exactly one vertex included at each t
        oper = npq.expand_gate(npq.IDTY, num_qubits, [0])
        for v in vertices_to_visit:
            oper += npq.expand_gate(-1 * x, num_qubits, [get_index(v, t)])
        hamiltonian += penalty_factor * (oper @ oper)
    for v in vertices_to_visit:  # add cost of edges from start point to start and end of path
        weight = get_weight(start_vertex, v)
        hamiltonian += npq.expand_gate(weight * x, num_qubits, [get_index(v, 0)])
        hamiltonian += npq.expand_gate(weight * x, num_qubits, [get_index(v, n - 1)])

    return hamiltonian, [start_vertex, *vertices_to_visit]


def get_graph_result(binary_number: str, n: int = 3):
    # returns the order in which to visit each of n vertices
    times = [binary_number[i * n: (i + 1) * n] for i in range(n)]
    visit_order = []
    for visit in times:
        these = [i for i, bit in enumerate(visit) if bool(int(bit))]
        if len(these) == 0:
            visit_order.append(None)
        elif len(these) == 1:
            visit_order.append(these[0])
        else:
            visit_order.append(these)
    return visit_order


def get_cost(binary_number: str):
    return get_hamiltonian(penalty_factor=0)[0][int(binary_number, 2), int(binary_number, 2)]


def get_energy(binary_number: str, penalty_factor: float):
    return get_hamiltonian(penalty_factor=penalty_factor)[0][int(binary_number, 2), int(binary_number, 2)]
