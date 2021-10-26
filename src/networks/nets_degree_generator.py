
import os
import pickle
import numpy as np
import numpy.random as npr

from multiprocessing import Pool, Lock
from multiprocessing.sharedctypes import RawArray

from generator import ba_graph_degree, ba_discrete_fitness_degree


def init_process(array_mem, array_shape):
    global process_array_mem, process_array_shape
    process_array_mem = array_mem
    process_array_shape = array_shape


def get_net(i):
    global n, m, gen, fv, lock
    avg_degree = np.frombuffer(process_array_mem, dtype=np.int64).reshape(process_array_shape)
    if args is not None:
        _, degree_hist, _ = gen(n, m, fitness_values=fv)
    else:
        _, degree_hist, _ = gen(n, m)
    # Aquire and release lock so only one process accessess the shared memory
    with lock:
        avg_degree += degree_hist


# Filepath
folder_path = os.path.join(os.path.abspath(''), 'Data', 'Generated', 'Barabasi')
folder_path = os.path.abspath(folder_path)

# Amount of networks to generate
nets = 1

# Network with n nodes each new node with m connections
n = 10000
m = 2

networks_generator = [
    ('Barabási–Albert', ba_graph_degree, None),
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.991, 0.223]),      # 0.768  Separation
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.223, 0.991]),
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.75, 0.25]),        # 0.5    Separation
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.25, 0.75]),
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.625, 0.375]),      # 0.25   Separation
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.375, 0.625]),
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.5625, 0.4375]),    # 0.125  Separation
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.4375, 0.5625]),
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.53125, 0.46875]),  # 0.0625 Separation
    # ('BAD Fitness', ba_discrete_fitness_degree, [0.46875, 0.53125]),
]

for model in networks_generator:
    # Extract data from tuple
    name = model[0]
    gen = model[1]
    args = model[2]
    print('Generating Model', name)
    data = {}

    if args is not None:
        fv = {i: value for i, value in enumerate(npr.choice(args, n))}
        fv[0] = args[0]
        data['fv'] = fv

    # avg_degree = np.zeros((n-m, n))
    # for i in tqdm(range(nets)):
    #     if args is not None:
    #         _, degree_hist, _ = gen(n, m, fitness_values=fv)
    #     else:
    #         _, degree_hist, _ = gen(n, m)
    #     avg_degree += degree_hist

    lock = Lock()
    array_mem = RawArray('l', (n-m)*n)
    avg_degree = np.frombuffer(array_mem, dtype=np.int64).reshape(n-m, n)

    # Make sure it inits in Zero
    np.copyto(avg_degree, np.zeros((n-m, n), dtype=np.int64))

    with Pool(initializer=init_process, initargs=(array_mem, (n-m, n)), processes=6) as p:
        p.map(get_net, range(nets))
    data['avg_degree'] = np.array(avg_degree)/nets

    # File Name
    filename = str(nets) + '_' + name.replace(' ', '') + f'_{args}_({n},{m}).dat'
    save_path = os.path.abspath(os.path.join(folder_path, filename))

    # Save the network for later analysis
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)