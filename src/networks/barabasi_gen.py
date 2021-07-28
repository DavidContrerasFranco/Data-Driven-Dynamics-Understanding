
import os
import networkx as nx
from tqdm import tqdm

# Filepath
base_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
save_path = 'Data-driven-Dynamics-Understanding/Data/Generated/Barabasi'

# Amount of networks to generate
nets = 10

# Generation Options
nodes = 10000
edges = 2

for net in tqdm(range(nets)):
    G = nx.generators.random_graphs.barabasi_albert_graph(nodes, edges)
    filename = str(net) + '_' + str(nodes) + 'N_' + str(edges) + 'E.gz'
    filepath = os.path.join(base_path, save_path, filename)
    nx.write_gpickle(G, filepath)