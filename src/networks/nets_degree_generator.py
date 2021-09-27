
import os
import numpy as np

from tqdm import tqdm

import generator


# Filepath
folder_path = os.path.join(os.path.abspath(''), '..', '..', 'Data', 'Generated', 'Barabasi')

# Amount of networks to generate
nets = 50

# Network with n nodes each new node with m connections
n = 10000
m = 2

# Generate Simple Barabasi w/ Degree
avg_degree = np.zeros((n, n))
for net in tqdm(range(nets), desc='BA Model        '):
    G, degree_hist, colors = generator.ba_degree(n, m)
    avg_degree += degree_hist
avg_degree /= nets
# File Name
save_path = os.path.join(folder_path, str(nets) + '_' + G.name + '.npy')
# Save the network for later analysis
with open(save_path, 'wb') as f:
    np.save(f, avg_degree)

# Generate Mixed Barabasi & Simple attachment w/ Degree
avg_degree = np.zeros((n, n))
for net in tqdm(range(nets), desc='Mixed Model     '):
    G, degree_hist, colors = generator.mixed_graph_degree(n, m)
    avg_degree += degree_hist
avg_degree /= nets
# File Name
save_path = os.path.join(folder_path, str(nets) + '_' + G.name + '.npy')
# Save the network for later analysis
with open(save_path, 'wb') as f:
    np.save(f, avg_degree)

# Generate Barabasi with Fitness w/ Degree
avg_degree = np.zeros((n, n))
for net in tqdm(range(nets), desc='BA Fitness Model'):
    G, degree_hist, colors = generator.ba_fitness_degree(n, m)
    avg_degree += degree_hist
avg_degree /= nets
# File Name
save_path = os.path.join(folder_path, str(nets) + '_' + G.name + '.npy')
# Save the network for later analysis
with open(save_path, 'wb') as f:
    np.save(f, avg_degree)