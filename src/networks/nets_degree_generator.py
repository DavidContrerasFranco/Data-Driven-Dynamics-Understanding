
import os
import copy
import time
import pickle
import ctypes
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt

from multiprocessing import Pool, Lock
from multiprocessing.sharedctypes import RawArray

from generator import ba_graph_degree, ba_discrete_fitness_degree


def init_process(array_mem, array_shape):
    global process_array_mem, process_array_shape
    process_array_mem = array_mem
    process_array_shape = array_shape


def get_net(i):
    global n, m, gen, fv, lock
    avg_degree = np.frombuffer(process_array_mem, dtype=np.uint32).reshape(process_array_shape)
    if args is not None:
        _, degree_hist, _ = gen(n, m, fitness_values=fv)
    else:
        _, degree_hist, _ = gen(n, m)
    # Aquire and release lock so only one process accessess the shared memory
    with lock:
        avg_degree += degree_hist


# Folder path
folder_path = os.path.join(os.path.abspath(''), 'Data', 'Generated', 'Barabasi')

# Amount of networks to generate
nets = 200

# Network with n nodes each new node with m connections
n = 10000
m = 2

# Init Multiprocessing Memory Usage
lock = Lock()
array_mem = RawArray(ctypes.c_uint32, (n-m)*n)

networks_generator = [
    ('Barabási–Albert', ba_graph_degree, None),
    ('BAD Fitness 0.991',   ba_discrete_fitness_degree, [0.991, 0.223]),      # 0.768  Separation
    ('BAD Fitness 0.223',   ba_discrete_fitness_degree, [0.223, 0.991]),
    ('BAD Fitness 0.75',    ba_discrete_fitness_degree, [0.75, 0.25]),        # 0.5    Separation
    ('BAD Fitness 0.25',    ba_discrete_fitness_degree, [0.25, 0.75]),
    ('BAD Fitness 0.625',   ba_discrete_fitness_degree, [0.625, 0.375]),      # 0.25   Separation
    ('BAD Fitness 0.375',   ba_discrete_fitness_degree, [0.375, 0.625]),
    ('BAD Fitness 0.5625',  ba_discrete_fitness_degree, [0.5625, 0.4375]),    # 0.125  Separation
    ('BAD Fitness 0.4375',  ba_discrete_fitness_degree, [0.4375, 0.5625]),
    ('BAD Fitness 0.53125', ba_discrete_fitness_degree, [0.53125, 0.46875]),  # 0.0625 Separation
    ('BAD Fitness 0.46875', ba_discrete_fitness_degree, [0.46875, 0.53125]),
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
        data['fv'] = list(fv.values())

    # Make sure it inits in Zero
    avg_degree = np.frombuffer(array_mem, dtype=np.uint32).reshape(n-m, n)
    np.copyto(avg_degree, np.zeros((n-m, n), dtype=np.uint32))

    tic = time.time()
    with Pool(initializer=init_process, initargs=(array_mem, (n-m, n)), processes=6) as p:
        p.map(get_net, range(nets))
    data['avg_degree'] = np.array(copy.deepcopy(avg_degree.tolist()))/nets
    toc = time.time()
    # File Name
    filename = str(nets) + '_' + name.replace(' ', '') + f'_{args}_({n},{m}).dat'
    save_path = os.path.abspath(os.path.join(folder_path, filename))

    # # Save the network for later analysis
    # with open(save_path, 'wb') as f:
    #     pickle.dump(data, f)
    print(toc - tic)

    plt.cla()
    plt.title(name + ' ' + str(toc - tic))
    plt.plot(data['avg_degree'][:,0])
    plt.savefig(os.path.abspath(os.path.join(folder_path, name + '.png')))
    # plt.show()