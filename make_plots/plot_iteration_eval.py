from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.colors as cols
from matplotlib.patches import  Rectangle
from matplotlib.lines import Line2D

import config
from tools.dataset import GHL_new
from tools.reader import load_GHL_data

cfg = config.Config()

EMB_DIM = 30
WINDOW_SIZE = 2400

# mpl.rc('font', family="Times New Roman")
mpl.rc('font', family="Times New Roman")  # Ten")
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.fontset'] = 'cm'
figwidth = 5.765
colors = [cols.to_rgb(color_name) for color_name in ['C0', 'C1', 'C2', 'C3']]

# %% First making the plot of the performance as a function of iterations

savefolder = cfg.dir.results

# Loading the results
results = pd.read_csv(savefolder.joinpath('results_multirep.csv'))
pr_results = results[results.curve == 'pr']
roc_results = results[results.curve == 'roc']

# Get the labels of the observations
X, L, _ = load_GHL_data(cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)
dataset = GHL_new(X, L, window_size=WINDOW_SIZE, stride_size=WINDOW_SIZE // 5,
                  obs_weights=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
labels = []
with torch.no_grad():
    for _, l, _ in tqdm(dataloader, total=len(dataloader)):
        labels.append(l)
labels = torch.concatenate(labels, dim=0)

# %% Load the z-space encodings
values = []
for iteration in tqdm(range(100)):
    values.append(torch.load(
        cfg.dir.results.joinpath(f'running_results/iteration_{iteration}/z-vals.pt'), map_location='cpu'))
values = torch.stack(values, dim=0)

# %% Plotting Mahalanobis distances as a function of iterations
from torch.linalg import inv
mean_vector = torch.mean(values, dim=1)
diff_vector = values - mean_vector[:, None, :]
covariances = torch.stack([inv(torch.cov(x.T)) for x in values])
ein1 = torch.einsum('ind,idq->inq', diff_vector, covariances)
z_mahal = torch.sqrt(torch.einsum('ind,ind->in', ein1, values))

# %% Grouping observations by degree of abnormality
c_val = labels.mean(dim=1)
cmap = plt.get_cmap('jet')

group_threshold = 0.5
group1 = z_mahal[:, c_val == 0]
group2 = z_mahal[:, torch.logical_and(c_val > 0, c_val < group_threshold)]
group3 = z_mahal[:, torch.logical_and(c_val > group_threshold, c_val < 1.0)]
group4 = z_mahal[:, c_val == 1.0]

percentiles = torch.tensor([0.25, 0.75])

fig, ax = plt.subplots(1, 1, figsize=(5.76*2, 3*2))
# Group 1
regions, lines = [], []
regions.append(ax.fill_between(torch.arange(100).numpy(), *torch.quantile(group1, percentiles, dim=1),
                               linestyle='dashed', facecolor=(*colors[0], 0.05), edgecolor=(*colors[0], 1.0)))
line, = ax.plot(group1.median(dim=1).values, linewidth=3, color='C0')
lines.append(line)
# Group 2
regions.append(ax.fill_between(torch.arange(100).numpy(), *torch.quantile(group2, percentiles, dim=1),
                               linestyle='dashed', facecolor=(*colors[1], 0.05), edgecolor=(*colors[1], 1.0)))
line, = ax.plot(group2.median(dim=1).values, linewidth=3, color='C1')  # , label='$0 < \\bullet \leq .5$')
lines.append(line)
# Group 3
regions.append(ax.fill_between(torch.arange(100).numpy(), *torch.quantile(group3, percentiles, dim=1),
                               linestyle='dashed', facecolor=(*colors[2], 0.05), edgecolor=(*colors[2], 1.0)))
line,  = ax.plot(group3.median(dim=1).values, linewidth=3, color='C2')  # , label='$.5 < \\bullet \leq 1.0$')
lines.append(line)
# Group 4
regions.append(ax.fill_between(torch.arange(100).numpy(), *torch.quantile(group4, percentiles, dim=1),
                               linestyle='dashed', facecolor=(*colors[3], 0.05), edgecolor=(*colors[3], 1.0)))
line,  = ax.plot(group4.median(dim=1).values, linewidth=3, color='C3')  # , label='1.0')
lines.append(line)

# Adding two legends
legend = ax.legend([(region, line) for region, line in zip(regions, lines)],
                   ['0', '$0 < \\bullet \leq .5$', '$.5 < \\bullet \leq 1.0$', '1.0'],
                   title='Fraction of anomalies', loc='upper left', framealpha=0.8)
line = Line2D([0], [0], color='black', linewidth=3)
patch = Rectangle([0, 0], 0.5, 0.5, edgecolor=(0, 0, 0, 0.5), facecolor=(0, 0, 0, 0.05), linestyle='dashed')
legend2 = ax.legend([line, patch], ['Median', '50% Conf. int.'], framealpha=0.8)
ax.add_artist(legend)

ax.set_ylabel('Mahalanobis distance')
ax.set_xlabel('Iteration')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
fig.savefig('plots/mahal_dist_increase_plot.png', dpi=400, bbox_inches='tight')
fig.show()

# %% Plot of the PR-AUCs of density and Mahalanobis distance as predictors in the normal space
z_mahal_pr = torch.tensor(pd.pivot(pr_results, columns=['rep'], index='iteration', values='z-mahal').values)
z_mahal_stats = torch.nanmean(z_mahal_pr, dim=1).repeat([3, 1]) + \
                torch.einsum('i,j->ij', torch.tensor([0, -1, 1]), torch.tensor(np.nanstd(z_mahal_pr, axis=1)))
density_pr = torch.tensor(pd.pivot(pr_results, columns=['rep'], index='iteration', values='Log-likelihood').values)
density_stats = torch.nanmean(density_pr, dim=1).repeat([3, 1]) + \
                torch.einsum('i,j->ij', torch.tensor([0, -1, 1]), torch.tensor(np.nanstd(density_pr, axis=1)))

# Amount of actual outliers
real_frac = ((labels.sum(dim=1) > 0)*1.0).mean()

fig, axes = plt.subplots(2, 1, figsize=(figwidth*2, 2*2), height_ratios=(2, 1))
# z-mahal
fill1 = axes[0].fill_between(np.arange(100), *z_mahal_stats[[1, 2]], facecolor=(*colors[0], 0.05),
                             edgecolor=(*colors[0], 1.0), linestyle='dashed')
line1, = axes[0].plot(z_mahal_stats[0], linewidth=3, color=colors[0])
# Density
fill2 = axes[0].fill_between(np.arange(100), *density_stats[[1, 2]], facecolor=(*colors[1], 0.05),
                     edgecolor=(*colors[1], 1.0), linestyle='dashed')
line2, = axes[0].plot(density_stats[0], linewidth=3, color=colors[1])

axes[0].set_ylabel('PR AUC')
legend = axes[0].legend([(line1, fill1), (line2, fill2)], ['Mahalanobis distance', 'Density'], title='Predictor',
                        loc='lower right', bbox_to_anchor=[0.965, 0.05])
line = Line2D([0], [0], linewidth=3, color='black')
rect = Rectangle([0, 0], 0.5, 0.5, facecolor=(0, 0, 0, 0.05), edgecolor=(0, 0, 0, 1.0), linestyle='dashed')
axes[0].legend([line, rect], ['Mean', r'$\pm$ s.d.'], loc='lower center', bbox_to_anchor=[0.57, 0.2])
axes[0].add_artist(legend)
axes[0].spines[['top', 'right', 'bottom']].set_visible(False)
axes[0].get_xaxis().set_visible(False)

# Fraction of weights that are 0
zero_weight = torch.tensor(pd.pivot(pd.read_csv(savefolder.joinpath('weight_frac.csv')),
                                    columns=['rep'], index='iteration').values)
zero_weight_stats = torch.mean(zero_weight, axis=1).repeat([3, 1]) + \
    torch.einsum('i,j->ij', torch.tensor([0, -1, 1]), torch.std(zero_weight, dim=1))
axes[1].fill_between(np.arange(100), *zero_weight_stats[[1, 2]])
axes[1].plot(zero_weight_stats[0], linewidth=3, label='Observed', color=colors[2])
axes[1].plot([0, 100], [real_frac, real_frac], color='black', linestyle='dashed', label='Actual')
axes[1].set_ylabel('Frac. of obs.\nwith zero weight')
axes[1].legend(loc='lower right')
axes[1].set_xlabel('Iteration')
axes[1].spines[['top', 'right', 'bottom']].set_visible(False)

fig.savefig('plots/auc_by_iteration.png', dpi=400, bbox_inches='tight')
fig.show()
