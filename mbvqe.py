import atexit
import json
import time
from datetime import datetime
from multiprocessing import Pool, Queue

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.optimize import minimize

import chemistry_package as chem
import np_quantum as npq
import variational_forms as vform

FOLDER_DIR = None
DATA_DIR = None
LOG_DIR = None
ENTROPY = None

NUM_RUNS = None
DISTANCES = None
HAMILTONIANS = None
PENALTY_TERM = None

ANSATZ_CIRCUIT = None
NUM_PARAMETERS = None

AVAILABLE_CORES = None

SUCCESS = False


def log_this(string: str):
    with open(LOG_DIR, 'a') as file:
        file.write(string)


def at_termination():
    # final logging
    runtime = time.time() - START_TIME
    log_this(f'Termination notice ({datetime.now()}):\n')
    if SUCCESS:
        log_this('   Job successful. All tasks closed.\n')
    else:
        log_this(f'   Premature exit! Tasks remain open.\n')
    log_this(f'\nTotal runtime: {int(runtime / 3600)} hours and {int((runtime % 3600) / 60)} minutes')


def minimise_energy(hamiltonian, initial_guess, writable):
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x},  # theta > 0
        {'type': 'ineq', 'fun': lambda x: 2 * np.pi - x}  # theta < 2*pi
    ]

    def minimise_func(angles):
        state = vform.mb_variational_form(ANSATZ_CIRCUIT, angles)
        energy = npq.expecth(hamiltonian, state)
        penalty = npq.expecth(PENALTY_TERM, state)
        writable.write(str(energy) + ', ')
        return energy + penalty

    writable.write('[')
    min_energy_angles = minimize(
        minimise_func,
        initial_guess,
        method='COBYLA',  # SLSQP, COBYLA
        # bounds=[(0, 2 * np.pi)] * len(initial_guess),  # only for SLSQP
        constraints=constraints,  # only for COBYLA
        options={
            'maxiter': 2000,
            'rhobeg': np.pi / 2
        }
    ).x
    min_energy_state = vform.mb_variational_form(ANSATZ_CIRCUIT, min_energy_angles)
    min_energy = npq.expecth(hamiltonian, min_energy_state)
    writable.write(str(min_energy) + ']')
    return min_energy_angles, min_energy


def vqe_single_run(file_name):
    energies = []
    current_parameters = RNG.random(NUM_PARAMETERS) * 2 * np.pi
    with open(FOLDER_DIR + file_name, 'w') as file:
        for hamiltonian in HAMILTONIANS:
            # current_parameters = RNG.random(NUM_PARAMETERS) * 2 * np.pi
            result = minimise_energy(hamiltonian, current_parameters, file)
            current_parameters = result[0]
            energies.append(result[1])
            file.write('\n')
    return energies


def worker_init(q):
    global RNG
    RNG = q.get()


def get_rng_queue(n, entropy=None):
    if entropy is None:
        entropy = int(time.time())
    seed_sequence = SeedSequence(entropy)
    seeds = seed_sequence.spawn(n)

    queue = Queue()
    for seed in seeds:
        queue.put(default_rng(seed))
    return queue


def main():
    print('Now running VQEs:')

    with open(DATA_DIR, 'w') as file:  # opening in write mode clears contents
        file.write('Evaluated distances:\n')
        file.write(json.dumps(DISTANCES.tolist()) + '\n')
        file.write(f'(interatomic distance [Å], energy [Hartree])\n')
    with open(LOG_DIR, 'w') as file:  # opening in write mode clears contents
        file.write(f'Log opened at {datetime.now()}\n\n'
                   f'Running VQE {NUM_RUNS} times for inter-atomic distances in range {interval} '
                   f'at {num_points} locations for a total of {NUM_RUNS * num_points} data points\n')
    # handle program termination
    atexit.register(at_termination)

    num_processes = min(AVAILABLE_CORES, NUM_RUNS)
    rng_q = get_rng_queue(num_processes, ENTROPY)
    pool = Pool(processes=num_processes, initializer=worker_init, initargs=(rng_q,))

    remaining_tasks = NUM_RUNS
    for result in pool.imap_unordered(vqe_single_run, [f'run_{run}.txt' for run in range(NUM_RUNS)]):
        with open(DATA_DIR, 'a') as file:
            file.write(json.dumps(result))
            file.write('\n')
        remaining_tasks -= 1
        print(f'    process closed ({remaining_tasks} remaining)')
        if remaining_tasks % num_processes == 0:
            elapsed_time = time.time() - START_TIME
            log_this(f'Status ({datetime.now()}):\n'
                     f'   Elapsed time (h:m): {int(elapsed_time / 3600)}:{int((elapsed_time % 3600) / 60)}\n'
                     f'   Tasks remaining: {int(remaining_tasks)}\n')
    pool.close()
    pool.join()
    print()
    print('Congratulations: Calculations done')


if __name__ == '__main__':
    FOLDER_DIR = '../data/tests/h2/'
    ENTROPY = 8844673

    NUM_RUNS = 3

    interval = (0.35, 2.5)
    num_points = 20
    num_entangling_steps = 1
    add_penalty_terms = False

    AVAILABLE_CORES = 3

    DISTANCES = np.linspace(*interval, num_points)
    HAMILTONIANS = [chem.h2_hamiltonian(distance) for distance in DISTANCES]
    PENALTY_TERM = 1 * np.linalg.matrix_power(0 * np.eye(2 ** 4) - chem.h2_spin_operator(), 2)
    PENALTY_TERM += 1 * np.linalg.matrix_power(2 * np.eye(2 ** 4) - chem.h2_number_operator(), 2)
    PENALTY_TERM *= add_penalty_terms

    ANSATZ_CIRCUIT = vform.get_vqe_circ(4, num_entangling_steps, vform.cnot_cyl_entangling)
    NUM_PARAMETERS = vform.number_of_parameters(ANSATZ_CIRCUIT)
    LOG_DIR = FOLDER_DIR + 'log.txt'
    DATA_DIR = FOLDER_DIR + 'results.txt'

    print()
    print(f'Running VQE {NUM_RUNS} times for interatomic distance in range {interval} at {num_points} locations')
    print()
    START_TIME = time.time()
    main()
    END_TIME = time.time()
    print(f'Finished in {round(END_TIME - START_TIME, 4)} seconds')
    print()
    SUCCESS = True
