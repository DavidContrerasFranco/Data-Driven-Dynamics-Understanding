
import os
import numpy as np

def load_networks(data_type):
    # Filepath
    base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    save_path = 'Data-driven-Dynamics-Understanding\\Data\\' + data_type
    data_path = os.path.join(base_path, save_path)

    # Get File Names
    networks_dir = []
    for file in os.listdir(data_path):
        networks_dir += [os.path.join(base_path, save_path, file)]
        
    return networks_dir

def get_degree_distribution(degrees):
    degrees = np.fromiter(dict(degrees).values(), dtype=int)
    nodes, counts = np.unique(degrees, return_counts=True)
    prob_dens = counts / np.sum(counts)

    return nodes, prob_dens