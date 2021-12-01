
import time
import copy
import numpy as np

from functools import reduce
from operator import indexOf, iconcat
from sysidentpy.utils.display_results import results
from sysidentpy.model_structure_selection import FROLS

def narmax_state_space(nx_model:FROLS, X_train, X_test, states_names):
    xlag = nx_model.xlag if type(nx_model.xlag) == int else max(reduce(iconcat, nx_model.xlag, []))
    max_lag = max(xlag, nx_model.ylag)
    narmax_model = {}
    narmax_time = 0
    coeffs = []
    regs = []
    sim = []

    for s_id, state in enumerate(states_names):
        model = copy.deepcopy(nx_model)

        x_train = np.delete(X_train, s_id, axis=1)[:-1]
        y_train = X_train[1:, s_id].reshape(-1, 1)

        x_test = np.delete(X_test, s_id, axis=1)
        y_test = X_test[0:nx_model.ylag, s_id].reshape(-1, 1)

        # Train model for one state and get time
        tic = time.time()
        model.fit(X=x_train, y=y_train)
        toc = time.time()
        narmax_time += toc - tic

        # Print resulting model
        param = results(model.final_model, model.theta, model.err, model.n_terms, dtype='sci')
        param_names = np.delete(states_names, s_id, axis=0).tolist()
        display_nx_model(param, model.theta, state, param_names, max_lag)

        # Simulate model for the state Z
        coeffs += [np.pad(model.theta.flatten(), (0, model.basis_function.__sizeof__() - len(model.theta)))]
        regs += [[list(eq) for eq in model.final_model]]
        sim += [model.predict(X=x_test, y=y_test)]

    # Stack results for models and predictions
    narmax_model['features'] = states_names
    narmax_model['coeffs'] = np.vstack(coeffs)
    narmax_model['regs'] = regs
    narmax_sim = np.hstack(sim)

    return narmax_model, narmax_sim, narmax_time


def display_nx_model(results, theta, output:str, input_vars, max_lag=1):
    regressors = ''
    for idx, regressor in enumerate(results):
        if idx > 0:
            regressors += ' +'
        for lag in range(0, max_lag):
            regressor[0] = regressor[0].replace('(k-' + str(lag + 1) + ')', '[k-' + str(lag) + ']')
        regressors += ' ' + f'{theta[idx][0]:.3E}' + ' ' + regressor[0].replace('-0', '')

    regressors = regressors.replace('y', output).replace('+ -', '- ')
    for idx, var in enumerate(input_vars):
        regressors = regressors.replace('x' + str(idx+1), var)

    print(output + '[k+1] =' + regressors)


def solution_to_regressors(sol_terms, feature_names, order):
    '''Convert an standard list of terms describing the solution into the NARMAX regressors format'''
    # TODO: Division with powers in conversion

    regressors = []
    for idx_eq, eq in enumerate(sol_terms):
        eq_regs = []
        eq_features = [feature_names[idx_eq]] + np.delete(feature_names, idx_eq).tolist()
        for term in eq:
            n = 0
            reg = np.zeros(order, dtype='int')
            split_terms = term.split()

            # Replace powers by repetitions
            new_split_terms = []
            for factor in split_terms:
                new_factor = [factor]
                if '^' in factor:
                    pos_power = indexOf(factor, '^')
                    new_factor = [factor[pos_power - 1]] * int(factor[pos_power + 1])
                elif '/' in factor:
                    pos_div = indexOf(factor, '/')
                    new_factor = [factor[pos_div - 1], factor[pos_div + 1]]
                new_split_terms += new_factor
                    
            # Convert terms into regressors
            for factor in new_split_terms:
                if factor.isalpha():
                    reg[n] = 1000*indexOf(eq_features, factor) + 1001
                    n += 1
                elif '_k-' in factor:
                    pos_k = indexOf(factor, 'k')
                    base = factor[:pos_k - 1]
                    delay =int(factor[pos_k + 2])
                    reg[n] = 1000*indexOf(eq_features, base) + 1000 + delay
                    n += 1

            eq_regs += [list(np.sort(reg))[::-1]]
        regressors += [eq_regs]
    return regressors