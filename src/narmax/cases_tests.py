
import os
import pickle
import numpy as np
import pysindy as ps
import subprocess as sp
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from sysidentpy.utils.display_results import results
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial

from utils import display_nx_model

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from networks.utils import barabasi_sol, barabasi_fitt, barabasi_diff

# Paths
source_folder = os.path.join(os.path.abspath(__file__), '..', '..', '..')
data_folder = os.path.abspath(os.path.join(source_folder, 'Data', 'Generated', 'Barabasi'))
plot_folder = os.path.join(source_folder, 'Reports', 'Figures', 'Barabasi', 'NARMAX')

# Model Characteristics
m = 2
n = 1e4
t = np.arange(1, n - m + 1)

k_ant_sol = barabasi_sol(m, 1, t)               # Degree evolution from the analytical solution
k_dif_sol = odeint(barabasi_diff, 2, t)[:,0]    # Degree evolution from differential equations

# Get Files
files = []
data_path = os.path.join(os.path.dirname( __file__ ), 'Recordings')
for filename in os.listdir(data_folder):
    if filename.endswith('.dat'):
        files +=[filename]

file:str
for file in files:
    print(file + ':')

    # Load Net Data
    file_path = os.path.abspath(os.path.join(data_folder, file))

    with open(file_path, "rb") as input_file:
        data = pickle.load(input_file)
    avg_degrees_hist = data['avg_degree']

    net_args = file.split('_')
    size = int(net_args[0])
    simple = eval(net_args[2])
    filename = '_'.join(net_args[1:]).replace('.dat', '')

    # Degree evolution from the analytical solution
    if simple is not None:
        fv = np.array(list(data['fv'].values()))
        # k_ant_sol = barabasi_fitt(m, fv, t, 1)
        k_ant_sol = barabasi_fitt(m, fv, avg_degrees_hist, 0)
    else:
        k_ant_sol = barabasi_sol(m, 1, t)

    # Simulated degree evolution of initial node
    k_model = avg_degrees_hist[:,0]


    cases = [
        {'name': 'Simple', 'degree': k_model, 'time': t, 'names' : ['t'],
         'model':FROLS(order_selection=True, info_criteria='bic', ylag=1, xlag=1,
                       estimator='least_squares', basis_function=Polynomial(degree=5))},

        {'name': 'Simple N_Info', 'degree': k_model, 'time': t, 'names' : ['t'],
         'model':FROLS(order_selection=True, info_criteria='bic', ylag=1, xlag=1, n_info_values=2,
                       estimator='least_squares', basis_function=Polynomial(degree=5))},

        {'name': 'Simple + Conformal Time', 'degree': k_model, 'time': 1/t, 'names' : ['1/t'],
         'model':FROLS(order_selection=True, info_criteria='bic', ylag=1, xlag=1,
                       estimator='least_squares', basis_function=Polynomial(degree=5))},

        {'name': 'Simple + Conformal Time N_Info', 'degree': k_model, 'time': 1/t, 'names' : ['1/t'],
         'model':FROLS(order_selection=True, info_criteria='bic', ylag=1, xlag=1, n_info_values=2,
                       estimator='least_squares', basis_function=Polynomial(degree=5))},
    ]

    for case in cases:
        nx_model:FROLS = case['model']
        name:str = case['name']

        if simple is not None:
            name = name.replace('Simple', 'Simple Fitness ' + str(simple[0]))
        if size == 1:
            name = name.replace('Simple', 'Single')

        degree = case['degree']
        time = case['time']
        var_name = case['names']

        # Train model for the state X and get time
        nx_model.fit(X=time.reshape(-1, 1), y=degree.reshape(-1, 1))
        
        # Get model parameters for the state X and print
        print('Case', name + ':')
        coeffs_y = np.pad(nx_model.theta.flatten(), (0, nx_model.basis_function.__sizeof__() - len(nx_model.theta)))
        params = results(nx_model.final_model, nx_model.theta, nx_model.err, nx_model.n_terms, dtype='sci')
        display_nx_model(params, nx_model.theta, 'k', var_name, 1)

        # Simulate the model through time
        narmax_sim = nx_model.predict(X=time.reshape(-1, 1), y=degree[0].reshape(-1, 1))
        
        # try:
        #     k_sindy = model.simulate(start, t=simul_t)
        # except ValueError:
        #     print('Overflow- Model grows too fast to simulate.\n')
        #     continue

        fig = plt.figure(figsize=(6,4), dpi= 200, facecolor='w', edgecolor='k')
        plt.plot(t, k_model, color='deepskyblue')
        plt.plot(t, narmax_sim, color='tab:red')
        plt.plot(t, k_ant_sol, color='darkolivegreen', linestyle='dashed')
        # plt.plot(t, k_dif_sol, color='yellowgreen', linestyle='dotted')
        plt.title('Degree Growth - ' + name)
        plt.xlabel('Time')
        plt.ylabel('Degree')
        plt.legend(['Simulation', 'NARMAX Estimation', 'Analytical Sol.', 'Diff. Eq. Sol.'])

        save_path = os.path.abspath(os.path.join(plot_folder, filename))
        try:
            if not os.path.isdir(save_path):
                sp.call(['mkdir', save_path])
        except Exception as e:
            pass
        plt.savefig(os.path.abspath(os.path.join(save_path, name + '.png')))
        # plt.show()
        print()