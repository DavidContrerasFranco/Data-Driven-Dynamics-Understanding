
import numpy as np
import networkx as nx

from utils import load_networks, get_degree_distribution

# Get File Names
networks_dir = load_networks('Generated\\Barabasi')
# net_dir = np.random.choice(networks_dir)
net_dir = networks_dir[16] # For debugging purposes
G = nx.read_gpickle(net_dir)

# Get degree distribution
nodes, prob_dens = get_degree_distribution(nx.degree(G))

# Degrees through time
degrees = np.fromiter(dict(nx.degree(G)).values(), dtype=int)
N = len(degrees)

degrees_t = np.zeros((N, N))
for num, t in enumerate(degrees_t):
    index = range(num+1)
    H = G.subgraph(index)
    degrees_sub = np.fromiter(dict(nx.degree(H)).values(), dtype=int)
    degrees_t[num] = np.pad(degrees_sub, (0, N - num - 1), 'constant', constant_values=0)

print(degrees_t)
