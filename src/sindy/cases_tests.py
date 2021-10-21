
import os
import numpy as np
import pysindy as ps
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

def barabasi_diff(k, t):
    """
    Returns the differential value of a node of degree value k at time t.

    Parameters:
    k : float = Current degree of the node
    t : int   = Current time

    Returns:
    float = Difference
    """
    return 0.5 * k / t


# Get File Name
file = '1_BADFitness_[0.991, 0.223]_(10000,2).npy'
source_folder = os.path.join(os.path.abspath(__file__), '..', '..', '..')
data_folder = os.path.join(source_folder, 'Data', 'Generated', 'Barabasi')
file_path = os.path.abspath(os.path.join(data_folder, file))
plot_folder = os.path.join(source_folder, 'Reports', 'Figures', 'Barabasi')

net_args = file.split('_')
size = int(net_args[0])
simple = eval(net_args[2])

# Model Characteristics
m = 2
n = 1e4
avg_degrees_hist = np.load(file_path)
t = np.arange(1, n - m + 1)

k_ant_sol = barabasi(2, 1, t)                   # Degree evolution from the analytical solution
k_dif_sol = odeint(barabasi_diff, 2, t)[:,0]    # Degree evolution from differential equations

# Simulated degree evolution of initial node
k_model = avg_degrees_hist[:,0]
k_mod_stck = np.stack((k_model, t), axis=-1)

# Custom library to add the degree differential
custom_library = ps.CustomLibrary(library_functions=[lambda x, y : x / y], 
                                  function_names=[lambda x, y : x + '/' + y])


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
    model : ps.SINDy = case['model']
    name = case['name']

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
        print('Overflow- Model grows too fast to simulate')
        continue

    if len(k_sindy.shape) > 1:
        k_sindy = k_sindy[:,0]

    fig = plt.figure(figsize=(6,4), dpi= 200, facecolor='w', edgecolor='k')
    plt.plot(t, k_model, color='deepskyblue')
    plt.plot(t, k_sindy, color='tab:red')
    plt.plot(t, k_ant_sol, color='darkolivegreen', linestyle='dashed')
    plt.plot(t, k_dif_sol, color='yellowgreen', linestyle='dotted')
    plt.title('Degree Growth - ' + name)
    plt.xlabel('Time')
    plt.ylabel('Degree')
    plt.legend(['Simulation', 'SINDy Estimation', 'Analytical Sol.', 'Diff. Eq. Sol.'])
    plt.savefig(os.path.abspath(os.path.join(plot_folder, name + '.png')))
    # plt.show()
    print()