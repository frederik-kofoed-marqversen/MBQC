import numpy as np

from np_quantum import IDTY, X, Y, Z, dagger

PAULIS = [-Z, -Y, -X, IDTY, X, Y, Z]


def psqrt(clifford_operator: np.ndarray) -> np.ndarray:
    # returns exp(1j * pi/4 * op) where op is pauli operator
    # also denoted sqrt(+ 1j * sigma)
    return (IDTY + 1j * clifford_operator) / np.sqrt(2)


def nsqrt(clifford_operator: np.ndarray) -> np.ndarray:
    # returns exp(-1j * pi/4 * op) where op is pauli operator
    # also denoted sqrt(- 1j * sigma)
    return (IDTY - 1j * clifford_operator) / np.sqrt(2)


def permutation_product(perm2: [int], perm1: [int]):
    # returns the result of perm2(perm1([1, 2, 3])
    return [int(np.sign(i) * perm2[abs(i) - 1]) for i in perm1]


def index_product(clifford_index2: int, clifford_index1: int):
    # returns the index of the product unitary clifford_index2 * clifford_index1
    return clifford_multiplication_table[clifford_index2][clifford_index1]


def index_inverse(clifford_index: int):
    return conjugate_index_table[clifford_index]


def permutation_inverse(perm: [int]):
    res = [0] * len(perm)
    for i in range(len(perm)):
        res[abs(perm[i]) - 1] = int(np.sign(perm[i])) * (i + 1)
    return res


# taken from "entanglement in graph states and its applications" - TABLE 1
operators = [
    IDTY, X, Y, Z,
    psqrt(X), nsqrt(X), Z @ psqrt(X), Z @ nsqrt(X),
    psqrt(Y), nsqrt(Y), Z @ psqrt(Y), Z @ nsqrt(Y),
    psqrt(Z), nsqrt(Z), X @ psqrt(Z), X @ nsqrt(Z),
    psqrt(Z) @ psqrt(X), psqrt(Z) @ nsqrt(X), nsqrt(Z) @ psqrt(X), nsqrt(Z) @ nsqrt(X),
    psqrt(Z) @ psqrt(Y), psqrt(Z) @ nsqrt(Y), nsqrt(Z) @ psqrt(Y), nsqrt(Z) @ nsqrt(Y),
]


# OBS! there are two mistakes in the table from the article.
# indexing from ZERO we have that permutation 17 and 18 must swap, and permutation 23 must be [-3, -1, 2]
# permutations2 = [
#     [1, 2, 3], [1, -2, -3], [-1, 2, -3], [-1, -2, 3],
#     [1, -3, 2], [1, 3, -2], [-1, -3, -2], [-1, 3, 2],
#     [3, 2, -1], [-3, 2, 1], [3, -2, 1], [-3, -2, -1],
#     [-2, 1, 3], [2, -1, 3], [2, 1, -3], [-2, -1, -3],
#     [-2, -3, 1], [-2, 3, -1], [2, -3, -1], [2, 3, 1],
#     [3, 1, 2], [-3, 1, -2], [3, -1, -2], [-3, -1, 2],
# ]

def np_index(list_to_search_in: list, array_to_find: np.ndarray) -> int:
    for i, element in enumerate(list_to_search_in):
        if np.allclose(element, array_to_find):
            return i
    raise ValueError('array not in list')


permutations = [
    [
        np_index(PAULIS, operator @ pauli @ dagger(operator)) - 3 for pauli in PAULIS[4:]
    ] for operator in operators
]

clifford_multiplication_table = [
    [
        permutations.index(permutation_product(perm2, perm1)) for perm1 in permutations
    ] for perm2 in permutations
]

conjugate_index_table = [
    permutations.index(permutation_inverse(perm)) for perm in permutations
]
