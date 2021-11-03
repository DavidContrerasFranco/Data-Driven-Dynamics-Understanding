
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


def barabasi_sol(m, t_i, t):
    """
    Returns the degree evolution of the Barabasi-Albert Model.

    Parameters:
    m   : int = Number of initial edges
    t_i : int = Time at which the node attached to the network with m edges.
    t   : int = Time vector for the range of time expected.

    Returns:
    ndarray = Degree evolution of the node attached at time t_i
    """
    return m * np.sqrt(t / t_i)


def barabasi_diff(k, t, m=2):
    """
    Returns the differential value of a node of degree value k at time t.

    Parameters:
    k : float = Current degree of the node
    t : int   = Current time

    Returns:
    float = Difference
    """
    return k / (m * t)