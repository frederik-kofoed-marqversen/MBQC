import numpy as np

import mbqgates
from mbqc_v2 import MBQCircuit, simulate_circuit


def cnot_cyl_entangling(vqe_circ: MBQCircuit):
    for i in range(vqe_circ.num_lines):
        vqe_circ.add_gate(mbqgates.cnot(), [i, (i + 1) % vqe_circ.num_lines])


def cnot_full_entangling(vqe_circ: MBQCircuit):  # NOT TESTED YET
    for int_dist in range(1, vqe_circ.num_lines // 2 + 1):
        for i in range(vqe_circ.num_lines // int_dist):
            vqe_circ.add_gate(mbqgates.cnot(), [i, (i + int_dist) % vqe_circ.num_lines])


def mb_entangling(vqe_circ: MBQCircuit):
    vqe_circ.add_gate(mbqgates.mb_entangler(vqe_circ.num_lines), list(range(vqe_circ.num_lines)))


def parametric_entangling(vqe_circ: MBQCircuit):
    vqe_circ.add_gate(mbqgates.cn_phase(0, vqe_circ.num_lines), list(range(vqe_circ.num_lines)))


def get_vqe_circ(num_lines, num_entangling_steps=1, entangling_function=cnot_full_entangling):
    vqe_circ = MBQCircuit(num_lines)

    for i in range(num_lines):
        vqe_circ.add_gate(mbqgates.hadamard(), [i])
        vqe_circ.add_gate(mbqgates.rx(0), [i])
        vqe_circ.add_gate(mbqgates.hadamard(), [i])
        vqe_circ.add_gate(mbqgates.rx(0), [i])
    for _ in range(num_entangling_steps):
        entangling_function(vqe_circ)
        for i in range(num_lines):
            vqe_circ.add_gate(mbqgates.euler_rotation(0, 0, 0), [i])

    # cleanup and pauli-reduction
    for input_qubit in vqe_circ.input_qubits:
        vqe_circ.measure_qubit_pauli(input_qubit, 'X', 0)
    vqe_circ.remove_pauli_measurements()
    return vqe_circ


def number_of_parameters(circuit):
    return len(circuit.measurement_sequence)


def mb_variational_form(circuit, angles: np.ndarray):
    if angles.size != number_of_parameters(circuit):
        raise Exception('number of given parameters not equal to number of rotations')
    angles = angles.flatten()
    for i, qubit in enumerate(circuit.measurement_sequence):
        circuit[qubit]['angle'] = angles[i]
    return simulate_circuit(circuit)
