
import os
import time
import networkx as nx
from tqdm import tqdm

import generator


# Generation Options
m = 2
n = 1000
reps = 10000

acum = 0
for _ in tqdm(range(reps)):
    tic = time.time()
    nx.generators.random_graphs.barabasi_albert_graph(n, m)
    toc = time.time()
    acum +=  toc - tic
print('Nx Package generator:\t', acum/reps)

# acum = 0
for _ in tqdm(range(reps)):
    tic = time.time()
    generator.mixed_graph_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('Mixed generator:\t', acum/reps)

acum = 0
for _ in tqdm(range(reps)):
    tic = time.time()
    generator.ba_fitness_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('BA-fitness generator:\t', acum/reps)

acum = 0
for _ in tqdm(range(reps)):
    tic = time.time()
    generator.ba_discrete_fitness_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('BA-fit.Dis. generator:\t', acum/reps)