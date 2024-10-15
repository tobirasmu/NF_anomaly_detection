import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
import numpy as np

from models.NF import MAF
from tools.reader import load_GHL_data
from tools.dataset import GHL
import config

cfg = config.Config()

EMB_DIM = 30
WINDOW_SIZE = 2400

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

mpl.rc('font', family="Times New Roman")
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.fontset'] = 'cm'
figwidth = 5.765

# %% GHL - Loading model, data, and embeddings
X, L, _ = load_GHL_data(cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)
dataset = GHL(X, L, window_size=WINDOW_SIZE, stride_size=WINDOW_SIZE // 5, obs_weights=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

save_folder = cfg.dir.results.joinpath('running_results')

# %% This block can be skipped if it has been run once.
normalizing_flow = MAF(n_blocks=32, input_size=EMB_DIM, hidden_size=256, n_hidden=1, cond_label_size=None,
                       batch_norm=False, activation='tanh')
normalizing_flow.load_state_dict(torch.load(save_folder.joinpath('nf_model_iteration_80.pt'), map_location='cpu'))
normalizing_flow.to(device)
normalizing_flow.eval()

embeddings = torch.load(save_folder.joinpath('embeddings.pt'), map_location='cpu')
z_vals, densities = [], []
with torch.no_grad():
    for i in tqdm(range(0, 24000, 1000)):
        z, det = normalizing_flow(embeddings[i:(i+1000)])
        z_vals.append(z)
        densities.append(torch.distributions.Normal(0, 1).log_prob(z) + det)

z_vals = torch.concatenate(z_vals, dim=0).cpu().detach()
densities = torch.concatenate(densities, dim=0).sum(dim=1).cpu().detach()

# %% Estimating the CDF of the density distribution
sampled_densities = []
with torch.no_grad():
    for i in tqdm(range(200)):
        x_rand = torch.randn((1000, EMB_DIM)).to(device)
        _, jacobian = normalizing_flow.inverse(x_rand)
        sampled_densities.append((torch.distributions.Normal(0, 1).log_prob(x_rand) - jacobian).sum(axis=1).detach().cpu())

estimated_density = torch.concatenate(sampled_densities)
estimated_density = torch.sort(estimated_density).values.cpu().detach().numpy()
estimated_probabilities = torch.linspace(1e-10, 1, 200000).numpy()

from tools.writer import save_pickle
save_pickle((embeddings, densities, z_vals, estimated_density, estimated_probabilities), 'make_plots/GHL_PCA_plot_data.pkl')

# %% Loading the saved data for easier plotting
from tools.reader import load_pickle
embeddings, densities, z_vals, estimated_density, estimated_probabilities = load_pickle('make_plots/GHL_PCA_plot_data.pkl')
interpolator = interp1d(estimated_density, estimated_probabilities, fill_value='extrapolate')

# %% Fitting the PCA
from sklearn.decomposition import PCA

n_z_vals = (z_vals - z_vals.mean(dim=0)) / z_vals.std(dim=0)
pca = PCA(n_components=30).fit(n_z_vals)
loadings = pca.transform(n_z_vals)
pca_rand = PCA(n_components=30).fit(torch.distributions.Normal(0, 1).sample(z_vals.shape))

fig, ax = plt.subplots(1, 1)
ax.plot(pca.explained_variance_)
ax.plot(pca_rand.explained_variance_, '--')
fig.show()

# %% Making the PCA and interpretation plots
group1 = [3431, 8892]
group2 = [17019, 17858]
group3 = [10397, 17246]
fig, ax = plt.subplots(1, 1, figsize=(5.765, 5.765))
ax.scatter(loadings[:, 0], loadings[:, 1], c='black', alpha=0.08)
ax.scatter(loadings[group1, 0], loadings[group1, 1], c='C0')
ax.scatter(loadings[group2, 0], loadings[group2, 1], c='C1')
ax.scatter(loadings[group3, 0], loadings[group3, 1], c='C2')
ax.set_xlabel(f'Principal component 1   ({pca.explained_variance_ratio_[0]*100:.2f} %)')
ax.set_ylabel(f'Principal component 2   ({pca.explained_variance_ratio_[1]*100:.2f} %)')
fig.savefig('plots/GHL_PCA.png', bbox_inches='tight', dpi=200)
fig.show()

num_extra_windows = 3
fig, axes = plt.subplots(3, 3, figsize=(5.765, 5.765))
color = ['C0', 'C1', 'C2']
for group_i, group in enumerate([group1, group2, group3]):
    for modem in group:
        x_plot = dataset.X[dataset.list_inds[modem]]
        x_window = x_plot[(dataset.start_inds[modem]):(dataset.start_inds[modem]+WINDOW_SIZE), :]
        xs_window = np.arange(num_extra_windows * WINDOW_SIZE, (num_extra_windows + 1) * WINDOW_SIZE)
        x_more = x_plot[(dataset.start_inds[modem]-WINDOW_SIZE*num_extra_windows):(dataset.start_inds[modem]+(num_extra_windows+1)*WINDOW_SIZE), :]
        xs_more = np.arange((2*num_extra_windows+1)*WINDOW_SIZE)

        axes[0, group_i].plot(xs_more, x_more[:, 0], c='black', alpha=0.5)
        axes[1, group_i].plot(xs_more, x_more[:, 1], c='black', alpha=0.5)
        axes[2, group_i].plot(xs_more, x_more[:, 2], c='black', alpha=0.5)

        axes[0, group_i].plot(xs_window, x_window[:, 0], c=color[group_i])
        axes[1, group_i].plot(xs_window, x_window[:, 1], c=color[group_i])
        axes[2, group_i].plot(xs_window, x_window[:, 2], c=color[group_i])

    axes[0, group_i].set_ylim(0, 3)
    axes[1, group_i].set_ylim(280, 340)
    axes[2, group_i].set_ylim(285, 355)
    if group_i > 0:
        axes[0, group_i].axis('off')
        axes[1, group_i].axis('off')
        axes[2, group_i].spines[['top', 'right', 'left']].set_visible(False)
        axes[2, group_i].set_yticks([])
        axes[2, group_i].set_xlabel('Time')
    else:
        axes[0, 0].spines[['top', 'bottom', 'right']].set_visible(False)
        axes[0, 0].set_xticks([])
        axes[1, 0].spines[['top', 'bottom', 'right']].set_visible(False)
        axes[1, 0].set_xticks([])
        axes[2, 0].spines[['top', 'right']].set_visible(False)
        axes[0, 0].set_ylabel('RT level')
        axes[1, 0].set_ylabel('RT temp')
        axes[2, 0].set_ylabel('HT temp')
        axes[2, 0].set_xlabel('Time')
fig.savefig('plots/GHL_PCA_clusters.png', bbox_inches='tight', dpi=200)
fig.show()
