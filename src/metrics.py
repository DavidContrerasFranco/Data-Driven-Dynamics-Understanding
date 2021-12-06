
import numpy as np
import pandas as pd

from pysindy import SINDy
from narmax.utils import solution_to_regressors
from sklearn.metrics import mean_squared_error
from sysidentpy.metrics import root_relative_squared_error, mean_forecast_error


def get_metrics_df(valid, sys_sol,  sindy_model:SINDy, narmax_model,
                                    sindy_sim,         narmax_sim,
                                    sindy_time,        narmax_time):
    ## Metrics
    metrics = {}
    sol_sparsity = [len(eq) for eq in sys_sol]

    # 1. CII
    # Both methods report a solution differently, therefore the subset search is done
    # differently
    basis_dict = sindy_model.feature_library.get_feature_names(sindy_model.feature_names)
    sprs_bool =[[feat in eq for feat in basis_dict] for eq in sys_sol]
    sindy_cii = np.all(np.logical_or(np.logical_not(sprs_bool), sindy_model.coefficients() != 0), axis=1)

    non_degree = len(narmax_model['regs'][0][0])
    sol_regs = solution_to_regressors(sys_sol, narmax_model['features'], non_degree)
    narmax_cii = [all([term in narmax_model['regs'][id_eq] for term in eq]) for id_eq, eq in enumerate(sol_regs)]

    metrics['CII'] = {'SINDy': sindy_cii, 'NARMAX': narmax_cii}

    # 2. COI
    sindy_coi = all(sindy_cii)
    narmax_coi = all(narmax_cii)
    metrics['COI'] = {'SINDy': sindy_coi, 'NARMAX': narmax_coi}

    # 3. L0 Norm
    sindy_l0 = np.linalg.norm(sindy_model.coefficients(), ord=0, axis=1)
    narmax_l0 = np.linalg.norm(narmax_model['coeffs'], ord=0, axis=1)
    metrics['L0 Norm'] = {'SINDy': sindy_l0.astype(int), 'NARMAX': narmax_l0.astype(int)}

    # 4. Sparsity (L0) Difference
    expected_sparsity = np.array(sol_sparsity)
    sindy_l0_diff = np.abs(expected_sparsity - sindy_l0)
    narmax_l0_diff = np.abs(expected_sparsity - narmax_l0)
    # TODO: Accuretely correct Difference for discrete time
    # sindy_l0_diff = np.abs(expected_sparsity - sindy_l0 - int(sindy_model.discrete_time))
    # narmax_l0_diff = np.abs(expected_sparsity - narmax_l0 - 1)
    metrics['L0 Norm Diff.'] = {'SINDy': sindy_l0_diff.astype(int), 'NARMAX': narmax_l0_diff.astype(int)}

    # 5. Complexity (Sum of L0)
    sindy_complexity = sindy_model.complexity
    narmax_complexity = np.sum(narmax_l0)
    metrics['Complexity'] = {'SINDy': int(sindy_complexity), 'NARMAX': int(narmax_complexity)}

    # 6. Forecast Error, Mean Squared Error (MSE), Root Relative Squared Error (RRSE)
    sindy_forecast = mean_forecast_error(valid, sindy_sim)
    sindy_mse = mean_squared_error(valid, sindy_sim)
    sindy_rrse = root_relative_squared_error(valid, sindy_sim)
    try:
        narmax_forecast = mean_forecast_error(valid, narmax_sim)
        narmax_mse = mean_squared_error(valid, narmax_sim)
        narmax_rrse = root_relative_squared_error(valid, narmax_sim)
    except ValueError:
        bounded = np.array([np.isfinite(narmax_sim)]).flatten()
        narmax_forecast = mean_forecast_error(valid[bounded], narmax_sim[bounded])
        narmax_mse = mean_squared_error(valid[bounded], narmax_sim[bounded])
        narmax_rrse = root_relative_squared_error(valid[bounded], narmax_sim[bounded])

    metrics['Forecast Error'] = {'SINDy': sindy_forecast, 'NARMAX': narmax_forecast}
    metrics['MSE'] = {'SINDy': sindy_mse, 'NARMAX': narmax_mse}
    metrics['RRSE'] = {'SINDy': sindy_rrse, 'NARMAX': narmax_rrse}

    # 8. Training Time
    metrics['Time'] = {'SINDy': sindy_time, 'NARMAX': narmax_time}

    # Save in a DataFrame and print
    return pd.DataFrame(metrics)