import atexit
import json
import time
from datetime import datetime
from multiprocessing import Pool, Queue

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.optimize import minimize

import np_quantum as npq
import variational_forms as vform
from spin_systems import spin_lattice_hamiltonian

# setup global variables

FOLDER_DIR = None
LOG_DIR = None
DATA_DIR = None
ENTROPY = None

NUM_RUNS = None
MAX_ITER = None
COUPLING_STRENGTHS = None
HAMILTONIANS = None
ANSATZ_CIRCUIT = None
NUM_PARAMETERS = None

AVAILABLE_CORES = None

START_TIME = None
SUCCESS = False


def log_this(string: str):
    with open(LOG_DIR, 'a') as file:
        file.write(string)


def write_data(string: str):
    with open(DATA_DIR, 'a') as file:
        file.write(string)


def at_termination():
    # final data write
    write_data(']')
    # final logging
    runtime = time.time() - START_TIME
    log_this(f'Termination notice ({datetime.now()}):\n')
    if SUCCESS:
        log_this('   Job successful. All tasks closed.\n')
    else:
        log_this(f'   Premature exit! Tasks remain open.\n')
    log_this(f'\nTotal runtime: {int(runtime / 3600)} hours and {int((runtime % 3600) / 60)} minutes')


def minimise_energy(hamiltonian, initial_guess):
    result = []
    constraints = [
        {'type': 'ineq', 'fun': lambda x: x},  # theta > 0
        {'type': 'ineq', 'fun': lambda x: 2 * np.pi - x}  # theta < 2*pi
    ]

    def get_variational_energy(angles):
        state = vform.mb_variational_form(ANSATZ_CIRCUIT, angles)
        energy = npq.expecth(hamiltonian, state)
        result.append(energy)
        return energy

    min_energy_angles = minimize(
        get_variational_energy,
        initial_guess,
        method='COBYLA',  # SLSQP, COBYLA
        # bounds=[(0, 2 * np.pi)] * len(initial_guess),  # only for SLSQP
        constraints=constraints,  # only for COBYLA
        options={
            'maxiter': MAX_ITER,
            'rhobeg': np.pi / 2
        }
    ).x
    get_variational_energy(min_energy_angles)
    return result


def vqe(index):
    initial_params = RNG.random(NUM_PARAMETERS) * 2 * np.pi
    result = minimise_energy(HAMILTONIANS[index], initial_params)
    return COUPLING_STRENGTHS[index], result[-1]


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

    # data file header
    with open(DATA_DIR, 'w') as file:  # opening in write mode clears contents
        file.write(f'(coupling strength [J/B], energy [a.u.])\n [')

    # logging
    print()
    print(f'Running VQE {NUM_RUNS} times for J/B in range {interval} at {num_points} locations')
    print()
    with open(LOG_DIR, 'w') as file:  # opening in write mode clears contents
        file.write(f'Log opened at {datetime.now()}\n\n'
                   f'Running VQE {NUM_RUNS} times for J/B in range {interval} at {num_points} locations '
                   f'for a total of {NUM_RUNS * num_points} separate runs\n')

    # handle program termination
    atexit.register(at_termination)

    # parallel processes setup
    rng_q = get_rng_queue(AVAILABLE_CORES, ENTROPY)
    pool = Pool(processes=AVAILABLE_CORES, initializer=worker_init, initargs=(rng_q,))

    # run processes
    task_indices = list(range(len(COUPLING_STRENGTHS))) * NUM_RUNS
    remaining_tasks = len(task_indices)
    last_status_time = START_TIME
    for result in pool.imap_unordered(vqe, task_indices):
        remaining_tasks -= 1

        # write data
        str_result = json.dumps(result)
        if remaining_tasks > 0:
            str_result += ','
        write_data(str_result)

        # logging
        print(f'    process closed ({remaining_tasks} remaining)')
        if time.time() - last_status_time > 3600:
            elapsed_time = time.time() - START_TIME
            log_this(f'Status ({datetime.now()}):\n'
                     f'   Elapsed time (h:m): {int(elapsed_time / 3600)}:{int((elapsed_time % 3600) / 60)}\n'
                     f'   Tasks remaining: {int(remaining_tasks)}\n')
            last_status_time = time.time()
    pool.close()
    pool.join()

    print()
    print('Job terminated! Calculations done')
    print()


if __name__ == '__main__':
    FOLDER_DIR = '../data/tests/spin_lattice/'
    ENTROPY = 816403

    NUM_RUNS = 10
    MAX_ITER = 2000

    num_points = 15
    num_entangling_steps = 2
    interval = (0, 1)
    n, m = 2, 2  # lattice size

    ANSATZ_CIRCUIT = vform.get_vqe_circ(n * m, num_entangling_steps, vform.parametric_entangling)

    AVAILABLE_CORES = 3
    DATA_DIR = FOLDER_DIR + 'results.txt'
    LOG_DIR = FOLDER_DIR + 'log.txt'
    COUPLING_STRENGTHS = np.linspace(interval[0], interval[1], num_points)
    HAMILTONIANS = [spin_lattice_hamiltonian(n, m, J, 1) for J in COUPLING_STRENGTHS]
    NUM_PARAMETERS = vform.number_of_parameters(ANSATZ_CIRCUIT)

    START_TIME = time.time()
    main()
    SUCCESS = True
