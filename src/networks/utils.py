
import os
import numpy as np

def load_networks(data_path):
    '''Get a list of paths for all the files inside data_path'''
    networks_dir = []
    for file in os.listdir(data_path):
        networks_dir += [os.path.join(data_path, file)]
        
    return networks_dir

def get_degree_distribution(degrees):
    degrees = np.fromiter(dict(degrees).values(), dtype=int)
    nodes, counts = np.unique(degrees, return_counts=True)
    prob_dens = counts / np.sum(counts)

    return nodes, prob_dens

import time
import ctypes
from tqdm import tqdm
from multiprocessing import Process, Pool
from multiprocessing.sharedctypes import RawArray


def init_process(array_mem, array_shape):
    global process_array_mem, process_array_shape
    process_array_mem = array_mem
    process_array_shape = array_shape


def f(i):
    arr = np.frombuffer(process_array_mem, dtype=np.float64).reshape(process_array_shape)
    this = (np.sum(np.mgrid[:n,:n], axis=0) + 1) ** i
    arr += this


n = 10000
m = 2
array_mem = RawArray(ctypes.c_double, n*n)
arr = np.frombuffer(array_mem, dtype=np.float64).reshape(n, n)
base = np.sum(np.mgrid[:n,:n], axis=0) + 1
print(base)

with Pool(initializer=init_process, initargs=(array_mem, arr.shape), processes=6) as p:
    p.map(f, range(10))
print(arr)