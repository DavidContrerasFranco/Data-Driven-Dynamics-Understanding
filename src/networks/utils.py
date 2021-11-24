
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


# Analytic solutions to the model

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


def barabasi_fitt(m, fv, k_h, t_i, c_star=1.255):
    """
    Returns the degree evolution of the Barabasi-Albert Model with fitness.

    Parameters:
    m   : int = Number of initial edges
    fv  : ndarray (R1) = Fitness values of all the nodes in the network
    k_h : ndarray (R2) = Degree evolution through time of the nodes.

    Returns:
    ndarray = Degree evolution of the node attached at time t_i
    """
    fit_evolve = k_h*fv
    sum_fit = np.sum(fit_evolve, axis=1)
    diff_k = m*(fit_evolve[:,t_i]/sum_fit)[:-1]
    return np.cumsum([m] + diff_k.tolist())
    # t = k_h
    # return m*((t / t_i) ** (fv[0]/c_star))
    # t = k_h
    # etas = np.array(list(set(fv)))
    # return m*((t / t_i) ** ((fv[0]*np.sum(etas))/(2*(np.prod(etas) + 1))))