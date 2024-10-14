import pandas as pd
import numpy as np

import config

cfg = config.Config()


save_folder = cfg.dir.results

# Loading the results
results = pd.read_csv(save_folder.joinpath('results_multirep.csv'))
pr_results = results[results.curve == 'pr']
roc_results = results[results.curve == 'roc']

# %% Returning the results for the different metrics:
print(f"{'Curve': ^5s}{'AUC': ^20s}{'Rec. Error': ^20s}{'Mahal. dist': ^20s}")
# ROC results
result_curve = roc_results
est_density = f"{result_curve['Log-likelihood'].mean(): ^5.3f} (pm {result_curve['Log-likelihood'].std(): ^4.3f})"
rec_error = f"{result_curve['MSE'].mean(): ^5.3f} (pm {result_curve['MSE'].std(): ^4.3f})"
mahal_dist = f"{result_curve['Mahalanobis'].mean(): ^5.3f} (pm {result_curve['Mahalanobis'].std(): ^4.3f})"
print(f"{'ROC': ^5s}{est_density: ^20s}", end='')
print(f"{rec_error: ^20s}", end='')
print(f"{mahal_dist: ^20s}")
# PR results
result_curve = pr_results
est_density = f"{result_curve['Log-likelihood'].mean(): ^5.3f} (pm {result_curve['Log-likelihood'].std(): ^4.3f})"
rec_error = f"{result_curve['MSE'].mean(): ^5.3f} (pm {result_curve['MSE'].std(): ^4.3f})"
mahal_dist = f"{result_curve['Mahalanobis'].mean(): ^5.3f} (pm {result_curve['Mahalanobis'].std(): ^4.3f})"
print(f"{'PR': ^5s}{est_density: ^20s}", end='')
print(f"{rec_error: ^20s}", end='')
print(f"{mahal_dist: ^20s}")
