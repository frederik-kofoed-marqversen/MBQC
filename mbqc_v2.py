import copy
import random
import warnings
from typing import Sequence

import numpy as np

import clifford_module as cliff
import np_quantum as qp
from graph_class_v2 import Graph
from graph_state_tn import GraphStateTensorNetwork


class MBQCError(Exception):
    """Base class for other exceptions related to MBQC"""
    pass


class LabellingError(MBQCError):
    pass


class StateVectorError(MBQCError):
    pass


class GateError(MBQCError):
    pass


class MissingAttributeError(MBQCError):
    pass


class GraphStateReductionModule:
    def __init__(self, graph: Graph, input_qubits: Sequence = (), output_qubits: Sequence = ()):
        self.graph = graph
        self.qubits = dict.fromkeys(self.graph.keys())
        self.input_qubits = list(input_qubits)
        self.output_qubits = list(output_qubits)

        self._set_qubits()

    def __getitem__(self, item):
        return self.qubits[item]

    def __iter__(self):
        return iter(self.qubits)

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        copied_instance = cls.__new__(cls)
        copied_instance.graph = copy.deepcopy(self.graph)
        copied_instance.qubits = copy.deepcopy(self.qubits)
        copied_instance.input_qubits = copy.deepcopy(self.input_qubits)
        copied_instance.output_qubits = copy.deepcopy(self.output_qubits)
        return copied_instance

    def _set_qubits(self):
        for index, qubit in enumerate(self):
            self.qubits[qubit] = {
                'LC-index': 0,
                'measurement_type': None,
                'measurement_result': None
            }
        return self

    def remove_redundant_qubits(self):
        qubits_to_remove = self.graph.get_isolated_vertices()
        self.delete_qubit(*qubits_to_remove)
        return self

    def delete_qubit(self, *qubits):
        for qubit in qubits:
            self.graph.delete_vertex(qubit)
            self.qubits.pop(qubit)
            if qubit in self.input_qubits:
                self.input_qubits.remove(qubit)
            if qubit in self.output_qubits:
                self.output_qubits.remove(qubit)
        return self

    def get_reduced_circuit(self):
        return copy.deepcopy(self).remove_redundant_qubits()

    def get_qubit_attributes(self, attr, qubits=None):
        # qubits must be a tuple of qubits
        if qubits is None:
            attribute_dict = {}
            for qubit in self.qubits:
                attribute_dict[qubit] = self[qubit][attr]
            return attribute_dict
        else:
            return [self[qubit][attr] for qubit in qubits]

    def get_clifford_index(self, qubits=None):
        return self.get_qubit_attributes('LC-index', qubits)

    def get_measurement_result(self, qubits=None):
        return self.get_qubit_attributes('measurement_result', qubits)

    def get_measurement_type(self, qubits=None):
        return self.get_qubit_attributes('measurement_type', qubits)

    def get_state_qubits(self):
        # returns all qubits yet to be measured
        return [qubit for qubit in self if self[qubit]['measurement_result'] is None]

    def measure_qubit_pauli(self, qubit, operator, measurement_result=None):
        # wrapper for the pauli measurement algorithm
        self[qubit]['measurement_type'] = qp.get_pauli_identifier(operator)
        self._efficient_pauli_measurement(qubit, qp.get_pauli_number(operator), measurement_result)
        return self

    def local_complementation_rule(self, qubit):
        # application of local graph complementation through |G> = U.dag()|tau(G)>
        # 4 corresponds to sqrt(+j * pi/4 * X) and 13 to sqrt(-j * pi/4 * Z)
        if qubit in self.input_qubits:
            warnings.warn('the local complementation rule does not apply to states not initially in the PLUS state')
        self._add_clifford_unitary(qubit, 4)
        for neighbour in self.graph.get_neighbours(qubit):
            self._add_clifford_unitary(neighbour, 13)
        self.graph.local_complementation(qubit)
        return self

    def _efficient_pauli_measurement(self, qubit, operator_number, measurement_result=None):
        if qubit in self.input_qubits:
            warnings.warn(
                'efficient simulation of pauli measurement not applicable to input states not in the PLUS state'
            )

        effective_operator = self._transformed_pauli_operator(qubit, operator_number)
        if measurement_result is None:
            if len(self.graph.get_neighbours(qubit)) == 0 and abs(effective_operator) == 1:
                # qubit initially in PLUS state so measurement result is 0 for X ~ 1, and 1 for -X ~ -1
                measurement_result = int((1 - effective_operator) / 2)
            else:
                # 50/50 chance between 0 and 1
                measurement_result = int(random.random() < 0.5)
        self[qubit]['measurement_result'] = measurement_result

        # The result of a negative pauli in projections is to flip the measurement result
        if effective_operator < 0:
            measurement_result = (measurement_result + 1) % 2
        self._apply_pauli_projection_rule(qubit, abs(effective_operator), measurement_result)

    def inverse_local_complementation_rule(self, qubit_index: int):
        # apply inverse unitary than in local_complementation_rule
        self._add_clifford_unitary(qubit_index, cliff.index_inverse(4))
        for neighbour in self.graph.get_neighbours(qubit_index):
            self._add_clifford_unitary(neighbour, cliff.index_inverse(13))
        self.graph.local_complementation(qubit_index)
        return self

    def _apply_pauli_projection_rule(self, qubit, pauli_operator_number, measurement_result):
        x, y, z = (qp.get_pauli_number(pauli) for pauli in ['X', 'Y', 'Z'])
        a = qubit
        b = None

        if pauli_operator_number == x:
            neighbours = self.graph.get_neighbours(a).difference(self.input_qubits)
            if len(neighbours) != 0:
                b = list(neighbours)[0]
                self.inverse_local_complementation_rule(b)
            else:
                self._add_clifford_unitary(a, cliff.index_inverse(13))
            pauli_operator_number = y

        if pauli_operator_number == y:
            self.inverse_local_complementation_rule(a)
            measurement_result = (measurement_result + 1) % 2
            pauli_operator_number = z

        if pauli_operator_number != z:
            raise MBQCError('operator has unexpected value ' + str(pauli_operator_number))

        if measurement_result == 1:
            for neighbour in self.graph.get_neighbours(a):
                self._add_clifford_unitary(neighbour, 3)

        self.graph.isolate_vertex(a)
        self[a]['LC-index'] = 0

        if b is not None:
            self.local_complementation_rule(b)

    def _transformed_pauli_operator(self, qubit, operator_number):
        # returns the pauli operator operator_index commuted through the LC-unitary at qubit_index
        # sigma * U = U * sigma', sigma' = U.dag() * sigma * U, return sigma'
        u_dagger_index = cliff.index_inverse(self[qubit]['LC-index'])
        return cliff.permutations[u_dagger_index][operator_number - 1]

    def _add_clifford_unitary(self, qubit, clifford_index):
        # adds a clifford unitary U2 to the graph state in front of the already present unitary: U|G> -> U*U2|G>
        self[qubit]['LC-index'] = cliff.index_product(self[qubit]['LC-index'], clifford_index)


class MBQComputation(GraphStateReductionModule):
    ADD_ONE = 'QZ6mJ42y'

    def __init__(self, graph: Graph, input_qubits: Sequence = (), output_qubits: Sequence = ()):
        super(MBQComputation, self).__init__(graph, input_qubits, output_qubits)
        self.byproduct_dependencies = [[[], []] for _ in range(len(input_qubits))]
        self.measurement_sequence = []

    def __deepcopy__(self, memodict={}):
        copied_instance = super(MBQComputation, self).__deepcopy__()
        copied_instance.byproduct_dependencies = copy.deepcopy(self.byproduct_dependencies)
        copied_instance.measurement_sequence = copy.deepcopy(self.measurement_sequence)
        return copied_instance

    def __getitem__(self, item):
        if item == self.ADD_ONE:
            return {'measurement_result': 1}
        else:
            return super(MBQComputation, self).__getitem__(item)

    def delete_qubit(self, *qubits):
        for qubit in qubits:
            if qubit in self.measurement_sequence:
                self.measurement_sequence.remove(qubit)
        # SHOULD ALSO DELETE QUBIT IN BYPRODUCT DEPENDENCY LIST AS WELL AS ALL MEASUREMENT DEPENDENCY LISTS
        # NOT CLEAR TO ME HOW THIS SHOULD BE DONE CORRECTLY
        super(MBQComputation, self).delete_qubit(*qubits)
        return self

    def set_pauli_measurement(self, qubit, pauli):
        self.qubits[qubit]['measurement_type'] = qp.get_pauli_identifier(pauli)
        return self

    def set_general_measurement(self, qubit, angle: float, dependencies: list):
        self.test_dependency_list(dependencies)
        self.qubits[qubit].update({
            'measurement_type': 'general',
            'angle': angle,
            'dependencies': dependencies
        })
        self.measurement_sequence.append(qubit)
        return self

    def get_measurement_state(self, qubit, assign_random_result=False):
        if assign_random_result:
            self[qubit]['measurement_result'] = int(random.random() < 0.5)
        if self[qubit]['measurement_result'] is None:
            raise MissingAttributeError('qubit has no measurement result')

        measurement_result = self[qubit]['measurement_result']
        measurement_type = self[qubit]['measurement_type']
        if measurement_type == 'general':
            parity = sum(self.get_measurement_result(self[qubit]['dependencies'])) % 2
            theta = (-1) ** parity * self[qubit]['angle']
            return qp.theta_state(theta, measurement_result)
        else:
            return qp.get_pauli_state(measurement_type, measurement_result)

    def do_pauli_measurements(self, measurement_result):
        # if measurement_result is set to 'rand' the pauli measurements will be performed as a random measurement
        if measurement_result == 'rand':
            measurement_result = None
        for qubit in self:
            if (self[qubit]['measurement_result'] is None
                    and qubit not in self.input_qubits
                    and qubit not in self.output_qubits):
                try:
                    self.measure_qubit_pauli(qubit, self[qubit]['measurement_type'], measurement_result)
                    # PauliOperatorError is raised if measurement_type is not a pauli measurement
                except qp.PauliOperatorError:
                    pass
        return self

    def remove_pauli_measurements(self, measurement_result=0):
        # if measurement_result is set to 'rand' the pauli measurements will be performed as a random measurement
        self.do_pauli_measurements(measurement_result)
        self.reduce_byproduct_dependencies()
        self.reduce_measurement_dependencies()
        self.reduce_measurement_sequence()
        self.remove_redundant_qubits()
        return self

    def add_qubits(self, circuit: GraphStateReductionModule):
        self.graph.union(circuit.graph)
        self.qubits.update(circuit.qubits)
        return self

    def identify_qubits(self, qubit1, qubit2, new_label=None):
        # NOTE: new qubit will inherit contents from qubit1
        if (self[qubit1]['measurement_result'] is not None) or (self[qubit2]['measurement_result'] is not None):
            raise LabellingError('identifying measured qubits not well defined')
        if new_label is None:
            new_label = qubit1

        new_contents = self.qubits[qubit1]
        self._relabel_qubit(qubit1, new_label)
        self._relabel_qubit(qubit2, new_label)
        self.qubits[new_label] = new_contents

        self.graph.identify_vertices(qubit1, qubit2, new_label)
        return self

    def relabel_qubit(self, current_label, new_label):
        if current_label == new_label:
            return self
        elif new_label in self:
            raise LabellingError(f'qubit labels must be unique. New label: {new_label} already exists')
        else:
            self.graph.relabel_vertex(current_label, new_label)
            self._relabel_qubit(current_label, new_label)
            return self

    def _relabel_qubit(self, current_label, new_label):
        # NOTE: this is not a safe method and may override existing qubits -> tests must be done beforehand
        # also note that no graph manipulations are done here
        # the mehtod handles the renaming of qubits in all attributes in self
        self.qubits[new_label] = self.qubits.pop(current_label)
        if current_label in self.input_qubits:
            self.input_qubits[self.input_qubits.index(current_label)] = new_label
        if current_label in self.output_qubits:
            self.output_qubits[self.output_qubits.index(current_label)] = new_label
        for dependencies in self.byproduct_dependencies:
            if current_label in dependencies[0]:
                dependencies[0][dependencies[0].index(current_label)] = new_label
            if current_label in dependencies[1]:
                dependencies[1][dependencies[1].index(current_label)] = new_label
        if current_label in self.measurement_sequence:
            self.measurement_sequence[self.measurement_sequence.index(current_label)] = new_label
        for qubit in self:
            if self[qubit]['measurement_type'] == 'general':
                if current_label in self[qubit]['dependencies']:
                    self[qubit]['dependencies'][self[qubit]['dependencies'].index(current_label)] = new_label

    def test_dependency_list(self, qubit_list):
        if any(qubit not in self and qubit != self.ADD_ONE for qubit in qubit_list):
            raise LabellingError('referenced a non-existent qubit in dependency list')
        if any(qubit in self.output_qubits for qubit in qubit_list):
            raise LabellingError('dependency on output qubits not well-defined')

    def set_byproduct_dependencies(self, output_index: int, x_dependencies: list, z_dependencies: list):
        self.test_dependency_list(x_dependencies + z_dependencies)
        self.byproduct_dependencies[output_index] = [x_dependencies, z_dependencies]
        return self

    def reduce_byproduct_dependencies(self):
        for dependencies in self.byproduct_dependencies:
            for i, qubits in enumerate(dependencies):
                # remove qubits that are already measured from list
                measurement_results = self.get_measurement_result(qubits)
                reduced_list = [qubits[i] for i in range(len(measurement_results)) if measurement_results[i] is None]
                # add 1 to byproduct if the measurements of the removed qubits sum to 1 mod 2
                parity = sum(measurement_result == 1 for measurement_result in measurement_results) % 2
                if parity == 1:
                    reduced_list.append(self.ADD_ONE)
                # remove every pair of equal qubits as they cancel mod 2
                temp = []
                for qubit in reduced_list:
                    if qubit in temp:
                        temp.remove(qubit)
                    else:
                        temp.append(qubit)
                reduced_list = temp
                # apply new reduced list
                dependencies[i] = reduced_list
        return self

    def get_byproduct(self):
        if any(measurement_result is None for measurement_result in
               self.get_measurement_result(qubit for qubit in self if qubit not in self.output_qubits)):
            raise MissingAttributeError(
                'cannot determine byproduct operator before all non-input/output qubits has been measured')
        result = []
        for dependencies in self.byproduct_dependencies:
            x_pow = sum(self.get_measurement_result(dependencies[0])) % 2
            z_pow = sum(self.get_measurement_result(dependencies[1])) % 2
            result.append([x_pow, z_pow])
        return result

    def get_byproduct_operator(self):
        return qp.tensor(*[
            np.linalg.matrix_power(qp.X, byproduct[0]) @ np.linalg.matrix_power(qp.Z, byproduct[1])
            for byproduct in self.get_byproduct()
        ])

    def reduce_measurement_dependencies(self):
        for qubit in self:
            if self[qubit]['measurement_type'] == 'general':
                dependencies = self[qubit]['dependencies']
                measurement_results = self.get_measurement_result(dependencies)
                self[qubit]['dependencies'] = [dependencies[i] for i in range(len(dependencies)) if
                                               measurement_results[i] is None]
                parity = sum(measurement_result == 1 for measurement_result in measurement_results) % 2
                if parity == 1:
                    self[qubit]['dependencies'].append(self.ADD_ONE)
        return self

    def reduce_measurement_sequence(self):
        for qubit in self.measurement_sequence:
            if self[qubit]['measurement_result'] is not None:
                self.measurement_sequence.remove(qubit)
        return self


class MBQGate(MBQComputation):
    def __init__(self, graph: Graph, input_qubits: Sequence, output_qubits: Sequence):
        super(MBQGate, self).__init__(graph, input_qubits, output_qubits)
        self.name = None
        self.dimension = len(input_qubits)
        self.propagate_byproduct = None  # fun(self, byproduct) -> new_byproduct, "and modifies self"

        for qubit in input_qubits:
            self.set_pauli_measurement(qubit, 'X')

    def __deepcopy__(self, memodict={}):
        copied_instance = super(MBQGate, self).__deepcopy__()
        copied_instance.name = copy.deepcopy(self.name)
        copied_instance.dimension = copy.deepcopy(self.dimension)
        copied_instance.propagate_byproduct = copy.deepcopy(self.propagate_byproduct)
        return copied_instance

    def add_identifier(self, identifier):
        for current_label in list(self.qubits.keys()):
            self.relabel_qubit(current_label, (identifier, current_label))
        return self


class MBQCircuit(MBQComputation):
    def __init__(self, num_information_lines):
        input_qubits = [f'I{i}' for i in range(num_information_lines)]
        super(MBQCircuit, self).__init__(Graph(input_qubits), input_qubits, copy.deepcopy(input_qubits))
        self.num_lines = num_information_lines
        self.num_gates = 0

    def __deepcopy__(self, memodict={}):
        copied_instance = super(MBQCircuit, self).__deepcopy__()
        copied_instance.num_lines = copy.deepcopy(self.num_lines)
        copied_instance.num_gates = copy.deepcopy(self.num_gates)
        return copied_instance

    def add_gate(self, gate: MBQGate, apply_to_lines: Sequence[int]):
        if len(apply_to_lines) != gate.dimension:
            raise GateError(
                f"number of qubits: {len(apply_to_lines)} does not match the gate dimension: {gate.dimension}"
            )
        # rename gate qubits to include gate numbering indexed from 0
        gate = copy.deepcopy(gate).add_identifier(self.num_gates)
        self.num_gates += 1

        # handle byproduct
        propagated_byproduct = gate.propagate_byproduct(gate, [self.byproduct_dependencies[i] for i in apply_to_lines])
        for i, line_index in enumerate(apply_to_lines):
            for operator in (0, 1):
                self.byproduct_dependencies[line_index][operator] = gate.byproduct_dependencies[i][operator] + \
                                                                    propagated_byproduct[i][operator]

        self.add_qubits(gate)
        self.measurement_sequence += gate.measurement_sequence
        # merge input of gate onto existing qubits
        for i, line in enumerate(apply_to_lines):
            # we keep the input names and discard the output names since outputs are never measured
            # and byproducts may depend on input measurement results
            self.identify_qubits(gate.input_qubits[i], self.output_qubits[line], new_label=gate.input_qubits[i])
            self.output_qubits[line] = gate.output_qubits[i]
        return self

    def get_gate_qubits(self, gate_num):
        return [qubit for qubit in self if qubit[0] == gate_num]


def simulate_circuit(circuit: MBQComputation, input_states: list = None):
    circuit = copy.deepcopy(circuit)
    tensor_network = GraphStateTensorNetwork(circuit.graph)

    # handling of input state
    if input_states is None:
        pass
    else:
        for qubit, state in zip(circuit.input_qubits, input_states):
            tensor_network.set_initial_state(qubit, state)

    # handling of local clifford
    for qubit in circuit:
        cliff_index = circuit[qubit]['LC-index']
        if cliff_index != 0:
            tensor_network.apply_gate(cliff.operators[cliff_index], [qubit])

    # handling of measurements
    for qubit in circuit:
        if qubit not in circuit.measurement_sequence and qubit not in circuit.output_qubits:
            tensor_network.project_qubit(qubit, circuit.get_measurement_state(qubit, True))
    for qubit in circuit.measurement_sequence:
        tensor_network.project_qubit(qubit, circuit.get_measurement_state(qubit, True))

    # calculation of state vector
    result = tensor_network.get_state(circuit.output_qubits)
    post_processing_operator = circuit.get_byproduct_operator()
    return post_processing_operator @ result
