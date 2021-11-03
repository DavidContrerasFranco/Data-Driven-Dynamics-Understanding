
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sysidentpy.metrics import root_relative_squared_error, mean_forecast_error

def get_metrics_df(valid, sol_sparsity, sindy_model, narmax_model,
                                        sindy_sim, narmax_sim,
                                        sindy_time, narmax_time):
    ## Metrics
    metrics = {}
    # 0. L0 Norm
    sindy_l0 = np.linalg.norm(sindy_model.coefficients(), ord=0, axis=1)
    narmax_l0 = np.linalg.norm(narmax_model, ord=0, axis=1)
    metrics['L0 Norm'] = {'SINDy': sindy_l0, 'NARMAX': narmax_l0}

    # 1. Sparsity (L0) Difference
    expected_sparsity = np.array(sol_sparsity)
    sindy_l0_diff = np.abs(expected_sparsity - sindy_l0)
    narmax_l0_diff = np.abs(expected_sparsity - narmax_l0)
    metrics['L0 Norm Diff.'] = {'SINDy': sindy_l0_diff, 'NARMAX': narmax_l0_diff}

    # 2. Complexity (Sum of L0)
    sindy_complexity = sindy_model.complexity
    narmax_complexity = np.sum(narmax_l0)
    metrics['Complexity'] = {'SINDy': sindy_complexity, 'NARMAX': narmax_complexity}

    # 5. Forecast Error
    sindy_forecast = mean_forecast_error(valid, sindy_sim)
    narmax_forecast = mean_forecast_error(valid, narmax_sim)
    metrics['Forecast Error'] = {'SINDy': sindy_forecast, 'NARMAX': narmax_forecast}

    # 3. Mean Squared Error (MSE)
    sindy_mse = mean_squared_error(valid, sindy_sim)
    narmax_mse = mean_squared_error(valid, narmax_sim)
    metrics['MSE'] = {'SINDy': sindy_mse, 'NARMAX': narmax_mse}

    # 6. Root Relative Squared Error (RRSE)
    sindy_rrse = root_relative_squared_error(valid, sindy_sim)
    narmax_rrse = root_relative_squared_error(valid, narmax_sim)
    metrics['RRSE'] = {'SINDy': sindy_rrse, 'NARMAX': narmax_rrse}

    # 4. R2 Score
    sindy_r2 = r2_score(valid, sindy_sim)
    narmax_r2 = r2_score(valid, narmax_sim)
    metrics['R2 Score'] = {'SINDy': sindy_r2, 'NARMAX': narmax_r2}

    # 5. Training Time
    metrics['Time'] = {'SINDy': sindy_time, 'NARMAX': narmax_time}

    # Save in a DataFrame and print
    return pd.DataFrame(metrics)