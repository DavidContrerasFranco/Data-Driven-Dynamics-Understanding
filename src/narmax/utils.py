
def display_nx_model(results, theta, variable:str):
    regressors = ''
    for idx, regressor in enumerate(results):
        if idx > 0:
            regressors += ' +'
        regressors += ' ' + f'{theta[idx][0]:.3E}' + ' ' + regressor[0]
    print(variable + '(k) =' + regressors)