
def display_nx_model(results, theta, output:str, input_vars):
    regressors = ''
    for idx, regressor in enumerate(results):
        if idx > 0:
            regressors += ' +'
        regressors += ' ' + f'{theta[idx][0]:.3E}' + ' ' + regressor[0]

    for idx, var in enumerate(input_vars):
        regressors = regressors.replace('x' + str(idx+1), var)


    print(output + '(k) =' + regressors.replace('y', output))