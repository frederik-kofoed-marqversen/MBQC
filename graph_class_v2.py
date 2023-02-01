import copy
from typing import Sequence


class GraphError(Exception):
    """
    Base class
    """
    pass


class VertexLabelError(GraphError):
    pass


class Graph(dict):
    def __init__(self, vertex_id_list=None, neighbours=None, edges=None):
        super(Graph, self).__init__()
        if vertex_id_list is not None:
            if isinstance(vertex_id_list, int):
                vertex_id_list = list(range(vertex_id_list))
            self.populate(vertex_id_list)
            if neighbours is not None:
                for i in range(len(self)):
                    self.add_neighbours(vertex_id_list[i], neighbours[i])
            if edges is not None:
                self.add_edge(*edges)

    def populate(self, vertex_id_list: Sequence):
        for vertex in vertex_id_list:
            if vertex in self:
                raise VertexLabelError(f'vertex id {vertex} already exists')
            self[vertex] = set()
        return self

    def add_vertex(self, vertex, neighbours=()):
        if vertex not in self:
            self[vertex] = set(neighbours)
            self[vertex].discard(vertex)
            for neighbour in neighbours:
                self[neighbour].add(vertex)
        else:
            raise VertexLabelError(f'vertex id {vertex} already exists')
        return self

    def add_edge(self, *vertex_two_tuples):
        for edge in vertex_two_tuples:
            if edge[0] == edge[1]:
                continue
            if (edge[0] not in self) or (edge[0] not in self):
                raise VertexLabelError('edge to undefined vertex')

            self[edge[0]].add(edge[1])
            self[edge[1]].add(edge[0])
        return self

    def add_neighbours(self, vertex, list_of_neighbours: Sequence):
        self[vertex].update(list_of_neighbours)
        for neighbour in list_of_neighbours:
            self[neighbour].add(vertex)
        return self

    def delete_vertex(self, *vertices):
        for vertex in vertices:
            for neighbour in self.pop(vertex):
                self[neighbour].remove(vertex)
        return self

    def delete_edge(self, *id_two_tuples):
        for edge in id_two_tuples:
            self[edge[0]].remove(edge[1])
            self[edge[1]].remove(edge[0])
        return self

    def get_incident_edges(self, vertex):
        return [(vertex, neighbour) for neighbour in self[vertex]]

    def isolate_vertex(self, *vertices):
        for vertex in vertices:
            self.delete_edge(*self.get_incident_edges(vertex))
        return self

    def relabel_vertex(self, current_label, new_label):
        if current_label == new_label:
            return self
        elif new_label in self:
            raise VertexLabelError(f'vertex labels must be unique. New label: {new_label} already exists')
        else:
            self[new_label] = self.pop(current_label)
            for vertex in self:
                if current_label in self[vertex]:
                    self[vertex].remove(current_label)
                    self[vertex].add(new_label)
            return self

    def identify_vertices(self, vertex1, vertex2, new_label):
        combined_neighbours = self[vertex1].union(self[vertex2])
        self.delete_vertex(vertex1, vertex2)
        self.add_vertex(new_label, combined_neighbours)
        return self

    def get_neighbours(self, vertex):
        return self[vertex]

    def get_edges(self, vertices=None):
        edges = set()
        if vertices is None:
            vertices = list(self.keys())
        for vertex in vertices:
            for neighbour in self[vertex]:
                if (neighbour, vertex) not in edges:
                    edges.add((vertex, neighbour))
        return edges

    def get_subgraph(self, vertex_list):
        raise Exception('get_subgraph method not yet implemented')

    def get_isolated_vertices(self):
        return [vertex for vertex in self if len(self[vertex]) == 0]

    def local_complementation(self, vertex):
        neighbours = self[vertex]
        for neighbour in neighbours:
            self[neighbour].symmetric_difference_update(neighbours)
            self[neighbour].discard(neighbour)
        return self

    def union(self, graph):
        graph = copy.deepcopy(graph)
        for vertex in graph:
            self[vertex] = graph[vertex]
        return self

    def get_inverse(self):
        inverse_graph = copy.deepcopy(self)
        vertices = list(inverse_graph.keys())
        for vertex in vertices:
            inverse_graph[vertex].symmetric_difference_update(vertices)
        return inverse_graph

    def get_complete(self):
        complete_graph = Graph()
        vertices = list(self.keys())
        for vertex in vertices:
            neighbours = vertices.copy()
            neighbours.remove(vertex)
            complete_graph[vertex] = neighbours
        return complete_graph

    def get_null_graph(self):
        # returns a copy of self with no edges
        return Graph(list(self.keys()))

    @classmethod
    def grid(cls, n: int, m: int):
        # returns grid graph with n rows and m columns
        grid = cls()
        for i in range(n):
            for j in range(m):
                neighbours = []
                if i > 0:
                    neighbours.append((i - 1, j))
                if j > 0:
                    neighbours.append((i, j - 1))
                grid.add_vertex((i, j), neighbours)
        return grid

    @classmethod
    def wire(cls, length: int):
        return cls(list(range(length)), edges=[(i - 1, i) for i in range(1, length)])

    @classmethod
    def disjoint_union(cls, graph0, graph1):
        vertices = list(graph0.keys()) + list(graph1.keys())
        edges = graph0.get_edges().union(graph1.get_edges())
        return cls(vertex_id_list=vertices, edges=edges)
