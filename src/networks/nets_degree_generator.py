
import os
import time
import ctypes
import numpy as np

from tqdm import tqdm
from multiprocessing import Process, Pool
from multiprocessing.sharedctypes import RawArray

import generator


def init_process(array_mem, array_shape):
    global process_array_mem, process_array_shape
    process_array_mem = array_mem
    process_array_shape = array_shape


def get_net(i):
    global n, m, gen, args
    avg_degree = np.frombuffer(process_array_mem, dtype=np.float64).reshape(process_array_shape)
    if args is not None:
        _, degree_hist, _ = gen(n, m, fitness_levels=args)
    else:
        _, degree_hist, _ = gen(n, m)
    avg_degree += degree_hist


# Filepath
folder_path = os.path.join(os.path.abspath(''), '..', '..', 'Data', 'Generated', 'Barabasi')
folder_path = os.path.abspath(folder_path)

# Amount of networks to generate
nets = 100

# Network with n nodes each new node with m connections
n = 10000
m = 2

networks_generator = [
    ('Barabási–Albert', generator.ba_graph_degree, None),
    ('BA Discrete Fitness', generator.ba_discrete_fitness_degree, [0.223, 0.991])
]

for model in networks_generator:
    # Extract data from tuple
    name = model[0]
    gen = model[1]
    args = model[2]
    print('Generating Model', name, end=": ")

    avg_degree = np.zeros((n-m, n), dtype='int64')
    for i in tqdm(range(nets)):
        if args is not None:
            _, degree_hist, _ = gen(n, m, fitness_levels=args)
        else:
            _, degree_hist, _ = gen(n, m)
        avg_degree += degree_hist

    # tic = time.time()
    # array_mem = RawArray(ctypes.c_double, (n-m)*n)
    # avg_degree = np.frombuffer(array_mem, dtype=np.float64).reshape(n-m, n)

    # with Pool(initializer=init_process, initargs=(array_mem, avg_degree.shape), processes=3) as p:
    #     p.map(get_net, range(nets))
    # name_file = name.replace(' ', '')
    # toc = time.time()
    # print(toc - tic)

    # File Name
    name_file = name.replace(' ', '')
    save_path = os.path.abspath(os.path.join(folder_path, str(nets) + '_' + name_file + f'_({n},{m}).npy'))

    # Save the network for later analysis
    with open(save_path, 'wb') as f:
        np.save(f, avg_degree)