
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