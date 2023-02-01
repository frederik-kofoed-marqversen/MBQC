import itertools

import numpy as np
from openfermion.hamiltonians import number_operator, s_squared_operator
from openfermion.transforms import get_fermion_operator, bravyi_kitaev
from openfermionpyscf import generate_molecular_hamiltonian
from openfermion import MolecularData
from openfermionpyscf import run_pyscf

from np_quantum import X, Y, Z, IDTY, tensor


def BK2numpy(qubit_hamiltonian, nr_qubits) -> np.ndarray:
    """
    OpenFermion to QuTiP translator -- by Christian Kraglund Andersen, ETH Zurich.
    Converts to QuTiP.
        qubit_hamiltonian (dict): 	Qubit Hamiltonian from OpenFermion.
        nr_qubits (int):        	Number of qubits in the circuit.

    Modified by Frederik K. Marqversen:
        Converts into Numpy rather than QuTiP
    """

    qubit_ops = {'X': X, 'Y': Y, 'Z': Z, 'I': IDTY}
    dict_hamiltonian = dict()
    for x in itertools.product(['I', 'X', 'Y', 'Z'], repeat=nr_qubits):
        key = tuple(((i, y) for i, y in enumerate(x) if y != 'I'))
        nicekey = tuple(y for y in x)

        if key in qubit_hamiltonian.terms.keys():
            operator = tensor(*[qubit_ops[y] for y in x])
            dict_hamiltonian[nicekey] = qubit_hamiltonian.terms[key] * operator

    keys = list(dict_hamiltonian.keys())
    numpy_hamiltonian = sum([dict_hamiltonian[key] for key in keys])

    return numpy_hamiltonian


def h2_hamiltonian(inter_atomic_distance: float) -> np.ndarray:
    geometry = [['H', [0, 0, 0]], ['H', [0, 0, inter_atomic_distance]]]
    multiplicity = 1
    # basis = 'cc - pvdz'
    basis = 'sto-3g'
    charge = 0
    molecular_hamiltonian = generate_molecular_hamiltonian(geometry, basis, multiplicity, charge)
    fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    encoded_hamiltonian = bravyi_kitaev(fermion_hamiltonian)
    qubit_hamiltonian = BK2numpy(encoded_hamiltonian, 4)
    return qubit_hamiltonian


def h2_exact_energies(inter_atomic_distance: float):
    return np.linalg.eigvalsh(h2_hamiltonian(inter_atomic_distance))


def h2_exact_e0(inter_atomic_distance: float):
    return min(h2_exact_energies(inter_atomic_distance))


def h2_spin_operator():
    return BK2numpy(bravyi_kitaev(s_squared_operator(4 // 2)), 4)


def h2_number_operator():
    return BK2numpy(bravyi_kitaev(number_operator(4)), 4)


def h2_sorted_energies(inter_atomic_distance: float):
    energies, states = np.linalg.eigh(h2_hamiltonian(inter_atomic_distance))
    states = [states[:, i] for i in range(len(energies))]

    number_operator = h2_number_operator()
    ion_energies = []
    non_ion_energies = []
    for i, state in enumerate(states):
        num_electrons = round(state @ number_operator @ state, 1)
        if num_electrons == 2:
            non_ion_energies.append(energies[i])
        elif num_electrons == 1:
            ion_energies.append(energies[i])
    return sorted(non_ion_energies)[:2] + sorted(ion_energies)[:1]


def h2_hartree_fock_energy(inter_atomic_distance: float):
    geometry = [['H', [0, 0, 0]], ['H', [0, 0, inter_atomic_distance]]]
    multiplicity = 1
    # basis = 'cc - pvdz'
    basis = 'sto-3g'
    charge = 0
    molecule = run_pyscf(MolecularData(geometry, basis, multiplicity, charge))
    return molecule.hf_energy
