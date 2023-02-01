import numpy as np


class PauliOperatorError(Exception):
    pass


class DimensionalError(Exception):
    pass


ZERO, ONE = np.array([1, 0]), np.array([0, 1])
PLUS, MINUS = np.array([1, 1]) / np.sqrt(2), np.array([1, -1]) / np.sqrt(2)
IPLUS, IMINUS = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)

IDTY = np.identity(2)
X, Y, Z = np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])
PAULIS = [X, Y, Z]

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

CZ = np.identity(4)
CZ[3, 3] = -1

CNOT = np.identity(4)
CNOT[2:, :] = np.flip(CNOT[2:, :], 0)


def get_pauli_number(pauli_identifier) -> int:
    if pauli_identifier in ['x', 'X', 1]:
        return 1
    elif pauli_identifier in ['y', 'Y', 2]:
        return 2
    elif pauli_identifier in ['z', 'Z', 3]:
        return 3
    else:
        raise PauliOperatorError(f'{pauli_identifier} is not a recognised pauli operator')


def get_pauli_identifier(pauli_identifier) -> str:
    return ['X', 'Y', 'Z'][get_pauli_number(pauli_identifier) - 1]


def get_pauli_operator(pauli_identifier) -> np.ndarray:
    return PAULIS[get_pauli_number(pauli_identifier) - 1]


def get_pauli_states(pauli_identifier):
    return [[PLUS, MINUS], [IPLUS, IMINUS], [ZERO, ONE]][get_pauli_number(pauli_identifier) - 1]


def get_pauli_state(pauli_identifier, state_index: int) -> np.ndarray:
    return get_pauli_states(pauli_identifier)[state_index]


def basis_state(N: int, index: int) -> np.ndarray:
    state = np.zeros(2 ** N)
    state[index] = 1
    return state


def theta_state(theta: float, index: int = 0) -> np.ndarray:
    return np.array([1, (-1) ** index * np.exp(1j * theta)])


def theta_states(theta: float):
    state = theta_state(theta)
    return [state, Z @ state]


def rotation(theta: float, vec) -> np.ndarray:
    return IDTY * np.cos(theta / 2) - 1j * sum(vec[i] * PAULIS[i] for i in range(3)) * np.sin(theta / 2)


def euler_rotation(theta1, theta2, theta3) -> np.ndarray:
    return rotation(theta3, [1, 0, 0]) @ rotation(theta2, [0, 0, 1]) @ rotation(theta1, [1, 0, 0])


def rx(theta: float) -> np.ndarray:
    return rotation(theta, [1, 0, 0])


def ket2dm(ket: np.ndarray) -> np.ndarray:
    if len(ket.shape) != 1:
        raise DimensionalError('state is not a ket')
    return np.outer(ket, np.conjugate(ket))


def normalise(ket: np.ndarray) -> np.ndarray:
    return ket / np.linalg.norm(ket)


def compare_kets(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(ket2dm(normalise(a)), ket2dm(normalise(b)))


def tensor(*arrays) -> np.ndarray:
    result = 1
    for array in arrays:
        result = np.kron(result, array)
    return result


def is_operator(oper: np.ndarray) -> bool:
    test1 = len(oper.shape) == 2
    test2 = oper.shape[0] == oper.shape[1]
    return test1 and test2


def is_vector(state: np.ndarray) -> bool:
    return len(state.shape) == 1


def is_hermitian(oper: np.ndarray) -> bool:
    return np.allclose(dagger(oper), oper)


def expect(oper: np.ndarray, state: np.ndarray):
    if not is_operator(oper) or not is_vector(state):
        raise DimensionalError()
    return np.conjugate(state) @ oper @ state


def expecth(oper: np.ndarray, state: np.ndarray):
    return expect(oper, state).real


def rand_ket(d=2) -> np.ndarray:
    return normalise(np.random.rand(d) + 1j * np.random.rand(d))


def dagger(array: np.ndarray) -> np.ndarray:
    return np.conjugate(array.T)


def _permute_tensor_product_rows(array: np.ndarray, new_ordering: list) -> np.ndarray:
    number_of_qubits = len(new_ordering)
    result = array.reshape((*[2] * number_of_qubits, -1))
    result = result.transpose([*new_ordering, number_of_qubits])
    result = result.reshape((2 ** number_of_qubits, -1))
    return result


def _permutation_inverse(perm):
    res = [0] * len(perm)
    for i in range(len(perm)):
        res[perm[i]] = i
    return res


def permute_tensor_product(array: np.ndarray, new_ordering) -> np.ndarray:
    if 2 ** len(new_ordering) != array.shape[0]:
        raise ValueError('new_ordering must be a permutation of all qubits')
    new_ordering = _permutation_inverse(new_ordering)

    result = _permute_tensor_product_rows(array, new_ordering)
    if len(array.shape) == 2:
        result = _permute_tensor_product_rows(result.T, new_ordering).T
    else:
        result = result.flatten()
    return result


def expand_gate(gate: np.ndarray, N: int, targets) -> np.ndarray:
    missing_indices = [i for i in range(N) if i not in targets]
    result = tensor(gate, *[IDTY] * len(missing_indices))
    result = permute_tensor_product(result, targets + missing_indices)
    return result


def num_qubits(gate: np.ndarray) -> int:
    return int(np.log2(gate.shape[0]))
