
import os
import pickle
import numpy as np
from numpy.lib.shape_base import _kron_dispatcher
import pysindy as ps
import subprocess as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Analytic approaches
def barabasi(m, t_i, t):
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

# Paths
source_folder = os.path.join(os.path.abspath(__file__), '..', '..', '..')
data_folder = os.path.abspath(os.path.join(source_folder, 'Data', 'Generated', 'Barabasi'))
plot_folder = os.path.join(source_folder, 'Reports', 'Figures', 'Barabasi', 'PySINDy')

# Model Characteristics
m = 2
n = 1e4
t = np.arange(1, n - m + 1)

k_ant_sol = barabasi(m, 1, t)                   # Degree evolution from the analytical solution
k_dif_sol = odeint(barabasi_diff, 2, t)[:,0]    # Degree evolution from differential equations

# Custom library to add the degree differential
custom_library = ps.CustomLibrary(library_functions=[lambda x, y : x / y], 
                                  function_names=[lambda x, y : x + '/' + y])


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
        fv = np.array(data['fv'])
        # k_ant_sol = barabasi_fitt(m, fv, t, 1)
        k_ant_sol = barabasi_fitt(m, fv, avg_degrees_hist, 0)
    else:
        k_ant_sol = barabasi(m, 1, t)

    print(k_ant_sol)

    # Simulated degree evolution of initial node
    k_model = avg_degrees_hist[:,0]
    k_mod_stck = np.stack((k_model, t), axis=-1)


    cases = [
        {'name': 'Simple', 'train': k_model, 'discrete': False,
        'model':ps.SINDy(differentiation_method=ps.FiniteDifference(),
                        feature_library=ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k'])},
        {'name': 'Simple + Time', 'train': k_mod_stck, 'discrete': False,
        'model':ps.SINDy(differentiation_method=ps.FiniteDifference(),
                        feature_library=ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k', 't'])},
        {'name': 'Simple + Time + CustomL', 'train': k_mod_stck, 'discrete': False,
        'model':ps.SINDy(differentiation_method=ps.FiniteDifference(),
                        feature_library=custom_library+ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k', 't'])},
        {'name': 'Simple + Time + CustomL Adjusted', 'train': k_mod_stck, 'discrete': False,
        'model':ps.SINDy(differentiation_method=ps.FiniteDifference(),
                        feature_library=custom_library+ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-2),
                        feature_names=['k', 't'])},
        {'name': 'Big Poly Simple', 'train': k_model, 'discrete': False,
        'model':ps.SINDy(differentiation_method=ps.FiniteDifference(),
                        feature_library=ps.PolynomialLibrary(degree=10),
                        optimizer=ps.STLSQ(threshold=1e-12),
                        feature_names=['k'])},
        {'name': 'Discrete Simple', 'train': k_model, 'discrete': True,
        'model':ps.SINDy(feature_library=ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k'],
                        discrete_time=True)},
        {'name': 'Discrete Simple + Time', 'train': k_mod_stck, 'discrete': True,
        'model':ps.SINDy(feature_library=ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k', 't'],
                        discrete_time=True)},
        {'name': 'Discrete Simple + Time + CustomL', 'train': k_mod_stck, 'discrete': True,
        'model':ps.SINDy(feature_library=custom_library+ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-5),
                        feature_names=['k', 't'],
                        discrete_time=True)},
        {'name': 'Discrete Simple + Time + CustomL Adjusted', 'train': k_mod_stck, 'discrete': True,
        'model':ps.SINDy(feature_library=custom_library+ps.PolynomialLibrary(degree=5),
                        optimizer=ps.STLSQ(threshold=1e-1),
                        feature_names=['k', 't'],
                        discrete_time=True)},
        {'name': 'Discrete Big Poly Simple', 'train': k_model, 'discrete': True,
        'model':ps.SINDy(feature_library=ps.PolynomialLibrary(degree=10),
                        optimizer=ps.STLSQ(threshold=1e-12),
                        feature_names=['k'],
                        discrete_time=True)},
    ]

    for case in cases:
        model:ps.SINDy = case['model']
        name:str = case['name']

        if simple is not None:
            name = name.replace('Simple', 'Simple Fitness ' + str(simple[0]))
        if size == 1:
            name = name.replace('Simple', 'Single')

        X = case['train']
        start = X[0]

        # Estimation without time awareness
        model.fit(X, quiet=True)

        # Print Model with name
        print('Case', name + ':')
        model.print()

        # Simulate the model through time
        simul_t = int(n-m) if case['discrete'] else t
        
        try:
            k_sindy = model.simulate(start, t=simul_t)
        except ValueError:
            print('Overflow- Model grows too fast to simulate.\n')
            continue

        if len(k_sindy.shape) > 1:
            k_sindy = k_sindy[:,0]

        fig = plt.figure(figsize=(6,4), dpi= 200, facecolor='w', edgecolor='k')
        plt.plot(t, k_model, color='deepskyblue')
        plt.plot(t, k_sindy, color='tab:red')
        plt.plot(t, k_ant_sol, color='darkolivegreen', linestyle='dashed')
        # plt.plot(t, k_dif_sol, color='yellowgreen', linestyle='dotted')
        plt.title('Degree Growth - ' + name)
        plt.xlabel('Time')
        plt.ylabel('Degree')
        plt.legend(['Simulation', 'SINDy Estimation', 'Analytical Sol.', 'Diff. Eq. Sol.'])

        save_path = os.path.abspath(os.path.join(plot_folder, filename))
        if not os.path.isdir(save_path):
            sp.call(['mkdir', save_path])
        plt.savefig(os.path.abspath(os.path.join(save_path, name + '.png')))
        # plt.show()
        print()