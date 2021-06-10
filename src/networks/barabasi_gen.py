
import os
import numpy as np
import networkx as nx

# Filepath
base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
save_path = 'Data-driven-Dynamics-Understanding\Data\Generated\Barabasi'

# Amount of networks to generate
nets = 20

# Generation Options
nodes = range(10, 50)
degrees = range(1, 5)

for net in range(nets):
    n = np.random.choice(nodes)
    d = np.random.choice(degrees)
    G = nx.generators.random_graphs.barabasi_albert_graph(n, d)
    filename = str(net) + '_' + str(n) + 'N_' + str(d) + 'D.gz'
    filepath = os.path.join(base_path, save_path, filename)
    nx.write_gpickle(G, filepath)