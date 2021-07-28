
import numpy as np
import networkx as nx
from tqdm import tqdm

from utils import load_networks, get_degree_distribution

# Get File Names
networks_dir = load_networks('Generated/Barabasi')
# net_dir = np.random.choice(networks_dir)
net_dir = networks_dir[0] # For debugging purposes
G = nx.read_gpickle(net_dir)

# Get degree distribution
nodes, prob_dens = get_degree_distribution(nx.degree(G))

# Degrees through time
degrees = np.fromiter(dict(nx.degree(G)).values(), dtype=int)
N = len(degrees)    # Total number of Nodes
L = 10              # Only the first L nodes are considered

degrees_t = np.zeros((N, L))
for num, _ in tqdm(enumerate(degrees_t)):
    index = range(num+1)
    H = G.subgraph(index)
    degrees_sub = np.fromiter(dict(nx.degree(H)).values(), dtype=int)
    if num < L:
        degrees_sub = np.pad(degrees_sub, (0, L - num - 1))
    degrees_t[num] = degrees_sub[:L]

print(degrees_t)
