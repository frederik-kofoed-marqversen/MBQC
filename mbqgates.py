import copy

from graph_class_v2 import Graph
from mbqc_v2 import MBQGate


def euler_rotation(theta1: float, theta2: float, theta3: float):
    # applies the gate Rx(theta3) * Rz(theta2) * Rx(theta1)
    angles = (theta1, theta2, theta3)
    gate = MBQGate(Graph.wire(5), [0], [4])
    gate.name = 'Euler_rotation'

    gate.set_general_measurement(1, angles[0], [0, gate.ADD_ONE])
    gate.set_general_measurement(2, angles[1], [1, gate.ADD_ONE])
    gate.set_general_measurement(3, angles[2], [0, 2, gate.ADD_ONE])

    gate.set_byproduct_dependencies(0, [1, 3], [0, 2])

    def propagate_function(rotation_gate: MBQGate, byproduct):
        byproduct = copy.deepcopy(byproduct)
        rotation_gate[rotation_gate.measurement_sequence[0]]['dependencies'] += byproduct[0][1]
        rotation_gate[rotation_gate.measurement_sequence[1]]['dependencies'] += byproduct[0][0]
        rotation_gate[rotation_gate.measurement_sequence[2]]['dependencies'] += byproduct[0][1]
        return byproduct

    gate.propagate_byproduct = propagate_function

    return gate


def hadamard_even_parity():
    gate = MBQGate(Graph.wire(5), [0], [4])
    gate.name = 'Hadamard'

    for i in range(1, 4):
        gate.set_pauli_measurement(i, 'Y')

    gate.set_byproduct_dependencies(0, [0, 2, 3], [1, 2])

    def propagate_function(hadamard_gate, byproduct):
        result = copy.deepcopy(byproduct)
        result[0][0], result[0][1] = result[0][1], result[0][0]
        return result

    gate.propagate_byproduct = propagate_function

    return gate


def rx(theta):
    # applies the gate Rx(theta3) * Rz(theta2) * Rx(theta1)
    gate = MBQGate(Graph.wire(3), [0], [2])
    gate.name = 'Z rotation'

    gate.set_general_measurement(1, theta, [0, MBQGate.ADD_ONE])
    gate.set_byproduct_dependencies(0, [1], [0])

    def propagate_function(rx_gate: MBQGate, byproduct):
        rx_gate[rx_gate.measurement_sequence[0]]['dependencies'] += byproduct[0][1]
        return byproduct

    gate.propagate_byproduct = propagate_function

    return gate


def hadamard():
    gate = MBQGate(Graph.wire(2), [0], [1])
    gate.name = 'Hadamard'

    gate.set_byproduct_dependencies(0, [0], [])

    def propagate_function(hadamard_gate, byproduct):
        result = copy.deepcopy(byproduct)
        result[0][0], result[0][1] = result[0][1], result[0][0]
        return result

    gate.propagate_byproduct = propagate_function

    return gate


def cnot():
    control, target = 0, 1
    gate = MBQGate(Graph.grid(3, 7), [(0, 0), (2, 0)], [(0, 6), (2, 6)])
    gate.name = 'CNOT'

    for col in [0, 1, 2, 4, 5, 6]:
        gate.measure_qubit_pauli((1, col), 'Z', 0)
    gate.remove_redundant_qubits()
    for qubit in [(0, i) for i in range(1, 6)] + [(1, 3), (2, 3)]:
        gate.set_pauli_measurement(qubit, 'Y')
    for col in [1, 2, 4, 5]:
        gate.set_pauli_measurement((2, col), 'X')

    gate.set_byproduct_dependencies(control, [(0, 1), (0, 2), (0, 4), (0, 5)],
                                    [(0, 0), (0, 2), (0, 3), (0, 4), (1, 3), (2, 0), (2, 2), MBQGate.ADD_ONE])
    gate.set_byproduct_dependencies(target, [(0, 1), (0, 2), (1, 3), (2, 1), (2, 3), (2, 5)], [(2, 0), (2, 2), (2, 4)])

    def propagate_function(c_not_gate, byproduct):
        result = copy.deepcopy(byproduct)
        result[control][1] += byproduct[target][1]
        result[target][0] += byproduct[control][0]
        return result

    gate.propagate_byproduct = propagate_function

    return gate


def cz():  # NOT TESTED YET
    gate = MBQGate(Graph.wire(2), [0, 1], [0, 1])
    gate.name = 'CZ'
    gate.set_byproduct_dependencies(0, [], [])
    gate.set_byproduct_dependencies(1, [], [])

    def propagate_function(cz_gate, byproduct):
        result = copy.deepcopy(byproduct)
        result[0][0] += byproduct[1][0]
        result[1][0] += byproduct[0][0]
        return result

    gate.propagate_byproduct = propagate_function
    return gate


def cn_phase(theta: float, n=1):
    if n < 2 or not isinstance(n, int):
        raise Exception(f'n must be an integer at least 2. n={n} was given.')
    graph = Graph(n)
    vertices = list(graph)
    graph.add_vertex('ancilla2', neighbours=vertices)
    graph.add_vertex('ancilla1', neighbours=['ancilla2'])

    gate = MBQGate(graph, vertices, vertices)
    gate.name = 'C^nPHASE()'

    gate.set_pauli_measurement('ancilla2', 'X')
    gate.set_general_measurement('ancilla1', theta, ['ancilla2', MBQGate.ADD_ONE])

    for i in range(len(vertices)):
        gate.set_byproduct_dependencies(i, [], ['ancilla1'])

    def propagate_function(cphase_gate: MBQGate, byproduct):
        for x_dep, _ in byproduct:
            cphase_gate[cphase_gate.measurement_sequence[0]]['dependencies'] += x_dep
        return byproduct

    gate.propagate_byproduct = propagate_function

    return gate


def mb_entangler(N):
    graph = Graph.grid(N, 2)
    for i in range(N - 1):
        graph.delete_edge([(i, 0), (i + 1, 0)])  # graph is now a comb-like structure
    graph.add_edge([(0, 1), (N - 1, 1)])  # wraps the graph to be a cylinder (entanglement also from bottom to top)

    input_qubits = [(i, 0) for i in range(N)]
    output_qubits = [(i, 1) for i in range(N)]
    gate = MBQGate(graph, input_qubits, output_qubits)
    gate.name = f'{N}-qubit entangler'

    for i, qubit in enumerate(input_qubits):
        gate.set_byproduct_dependencies(i, [], [qubit])

    def propagate_function(entangler_gate, byproduct):
        result = copy.deepcopy(byproduct)
        for i in range(N):
            result[i][0] = byproduct[i][1]
            result[i][1] = byproduct[i][0] + byproduct[(i - 1) % N][1] + byproduct[(i + 1) % N][1]
        return result

    gate.propagate_byproduct = propagate_function
    gate.byproduct_dependencies = propagate_function(gate, gate.byproduct_dependencies)

    return gate
