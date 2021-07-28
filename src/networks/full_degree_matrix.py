
import os
import numpy as np
import networkx as nx

from tqdm import tqdm

from utils import load_networks, get_degree_distribution

# Get File Names
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'Data')
networks_dir = load_networks(os.path.join(data_path, 'Generated', 'Barabasi'))

for net_dir in networks_dir:
    print('Calculating Matrix degree for', os.path.basename(net_dir))
    G = nx.read_gpickle(net_dir)
    N = G.number_of_nodes()

    # Degrees through time
    degrees_t = np.zeros((N, N))
    for num, t in tqdm(enumerate(degrees_t)):
        index = range(num+1)
        H = G.subgraph(index)
        degrees_sub = np.fromiter(dict(nx.degree(H)).values(), dtype=int)
        # degrees_sub = list(dict(nx.degree(H)).values())
        degrees_t[num] = np.pad(degrees_sub, (0, N - num - 1), 'constant', constant_values=0)

    with open(net_dir.replace('.gz', '.npy'), 'wb') as f:
        np.save(f, degrees_t)