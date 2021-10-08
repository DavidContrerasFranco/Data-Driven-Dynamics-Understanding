
import os
import time
import networkx as nx
from tqdm import tqdm

import generator


# Generation Options
m = 2
n = 1000
reps = 100

acum = 0
for i in range(reps):
    tic = time.time()
    G = nx.generators.random_graphs.barabasi_albert_graph(n, m)
    toc = time.time()
    acum +=  toc - tic
print('Nx Package generator:\t', acum/reps)

acum = 0
for i in range(reps):
    tic = time.time()
    G, _, _ = generator.ba_graph_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('BA w/ degree generator:\t', acum/reps)

acum = 0
for i in range(reps):
    tic = time.time()
    G, _, _ = generator.mixed_graph_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('Mixed generator:\t', acum/reps)

acum = 0
for i in range(reps):
    tic = time.time()
    G, _, _ = generator.ba_fitness_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('BA-fitness generator:\t', acum/reps)

acum = 0
for i in range(reps):
    tic = time.time()
    G, _, _ = generator.ba_discrete_fitness_degree(n, m)
    toc = time.time()
    acum +=  toc - tic
print('BA-fit.Dis. generator:\t', acum/reps)