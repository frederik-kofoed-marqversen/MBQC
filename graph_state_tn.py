import numpy as np
import tensornetwork as tn

import np_quantum as npq
from graph_class_v2 import Graph


class GraphStateTensorNetwork:
    def __init__(self, graph: Graph):
        self.initial_state_nodes = dict()
        self.qubit_edges = dict()

        self._build_graph_state_tensor_network(graph)

    def _build_graph_state_tensor_network(self, graph: Graph):
        self.initial_state_nodes = {name: tn.Node(npq.PLUS) for name in graph}
        self.qubit_edges = {name: node[0] for name, node in self.initial_state_nodes.items()}
        for edge in graph.get_edges():
            # self.apply_gate(npq.CZ, edge)
            self.add_cz(*edge)

    def set_initial_state_by_edge(self, qubit_name, edge):
        if self.initial_state_nodes[qubit_name].get_edge(0).is_dangling():
            self.initial_state_nodes[qubit_name] = tn.get_all_nodes([edge])
            self.qubit_edges[qubit_name] = edge
        else:
            dangling_edge = tn.remove_node(self.initial_state_nodes.pop(qubit_name))[1][0]
            edge ^ dangling_edge
        return self

    def set_initial_state(self, qubit_name, state: np.ndarray):
        if not len(state) == 2:
            raise Exception(f' {state} is not a qubit state')
        self.set_initial_state_by_edge(qubit_name, tn.Node(state)[0])
        return self

    def get_state(self, qubit_output_order: list = None) -> np.ndarray:
        if qubit_output_order is None:
            edge_order = list(self.qubit_edges.values())
        else:
            edge_order = [self.qubit_edges[qubit_name] for qubit_name in qubit_output_order]

        # MAYBE NEED SOME COPYING PROCEDURE AS TO NOT DESTROY NETWORK AFTER CALCULATING STATE
        nodes = tn.reachable(self.qubit_edges.values())
        final_node = tn.contractors.greedy(nodes, output_edge_order=edge_order)
        tn.flatten_edges(final_node.edges)
        state = final_node.tensor
        return npq.normalise(state)

    def memory_usage(self) -> int:
        return sum(node.tensor.nbytes for node in tn.reachable(self.initial_state_nodes.values()))

    def apply_gate(self, gate: np.ndarray, operating_qubit_names):
        gate_node = tn.Node(gate)
        num_qubits = len(operating_qubit_names)

        input_edges = [gate_node[1]]
        output_edges = [gate_node[0]]
        if num_qubits > 1:
            input_edges = tn.split_edge(input_edges[0], tuple(2 for _ in range(num_qubits)))
            output_edges = tn.split_edge(output_edges[0], tuple(2 for _ in range(num_qubits)))

        for qubit_name, input_edge, output_edge in zip(operating_qubit_names, input_edges, output_edges):
            self.qubit_edges[qubit_name] ^ input_edge
            self.qubit_edges[qubit_name] = output_edge
        return self

    def add_cz(self, qubit1, qubit2):
        tensor = np.array(
            [
                [[1, 0], [0, 1]],
                [[0, 0], [0, np.sqrt(2) * 1j]],
            ]
        )
        node1, node2 = tn.Node(tensor), tn.Node(tensor)
        node1[0] ^ node2[0]
        self.qubit_edges[qubit1] ^ node1[1]
        self.qubit_edges[qubit2] ^ node2[1]
        self.qubit_edges[qubit1] = node1[2]
        self.qubit_edges[qubit2] = node2[2]
        return self

    def project_qubit(self, qubit_name, state: np.ndarray):
        self.qubit_edges.pop(qubit_name) ^ tn.Node(np.conjugate(state))[0]
        return self
