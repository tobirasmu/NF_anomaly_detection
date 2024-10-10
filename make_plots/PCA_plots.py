import os
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d
import numpy as np

from tools.reader import load_GHL_data
from tools.dataset import GHL_new
from models.AE import auto_encoder
from models.NF import MAF
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

# %% GHL - Loading models and data
iteration = 80

save_folder = cfg.dir.data.joinpath(f'video_experiment_1perc/rep1/iteration_{iteration}')
logLiks = torch.load(save_folder.joinpath('log-likelihoods.pkl'), map_location='cpu')

autoencoder = auto_encoder(5, kernel_size=5, channels=256, emb_dim=EMB_DIM, window_size=WINDOW_SIZE, variational=False)
autoencoder.load_state_dict(torch.load(save_folder.parent.joinpath('AE_model.pt'), map_location='cpu'))
autoencoder.to(device)
autoencoder.eval()

normalizing_flow = MAF(n_blocks=32, input_size=EMB_DIM, hidden_size=256, n_hidden=1, cond_label_size=None,
                       batch_norm=False, activation='tanh')
normalizing_flow.load_state_dict(torch.load(save_folder.joinpath('nf_model.pt'), map_location='cpu'))
normalizing_flow.to(device)
normalizing_flow.eval()

X, L, _ = load_GHL_data(cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)
dataset = GHL_new(X, L, window_size=WINDOW_SIZE, stride_size=WINDOW_SIZE // 5, obs_weights=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

observation_weights = torch.load(save_folder.joinpath('observation_weights.pt'), map_location='cpu')

# %% Encoding and calculating the densities
with torch.no_grad():
    embeddings, labels, densities, z_vals = [], [], [], []
    for batch, label, _ in tqdm(dataloader):
        enc, _ = autoencoder(batch.to(device))
        z, det = normalizing_flow(enc)

        embeddings.append(enc)
        densities.append(torch.distributions.Normal(0, 1).log_prob(z) + det)
        z_vals.append(z)
        labels.append(label)

embeddings = torch.concatenate(embeddings, dim=0).cpu().detach()
densities = torch.concatenate(densities, dim=0).sum(dim=1).cpu().detach()
z_vals = torch.concatenate(z_vals, dim=0).cpu().detach()
labels = (torch.concatenate(labels, dim=0).sum(dim=1) > 0) * 1.0

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
save_pickle((embeddings, densities, z_vals, estimated_density, estimated_probabilities, labels), 'analysis/make_plots/GHL_PLOT_DATA.pkl')

# %% Loading the saved data for easier plotting
from tools.reader import load_pickle
embeddings, densities, z_vals, estimated_density, estimated_probabilities, labels = load_pickle('analysis/make_plots/GHL_PLOT_DATA.pkl')
interpolator = interp1d(estimated_density, estimated_probabilities, fill_value='extrapolate')

# %%
fig, axes = plt.subplots(22, 22, figsize=(30, 30))
axes = axes.reshape(-1)
plot_idx = 0
for i in range(30):
    for j in range(i + 1, 30):
        axes[plot_idx].scatter(z_vals[labels == 1, i], z_vals[labels == 1, j], c='red')
        axes[plot_idx].scatter(z_vals[labels == 0, i], z_vals[labels == 0, j], c='black')
        axes[plot_idx].set_xlabel(i)
        axes[plot_idx].set_ylabel(j)
        plot_idx += 1
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.scatter(z_vals[:, 6], z_vals[:, 10], c=interpolator(densities), cmap='inferno')
# ax.scatter(z_vals[labels == 1, 6], z_vals[labels == 1, 10], c='red')
# ax.scatter(z_vals[labels == 0, 6], z_vals[labels == 0, 10], c='black')
fig.show()

# %% Trying me some PCAing
from sklearn.decomposition import PCA

n_z_vals = (z_vals - z_vals.mean(dim=0)) / z_vals.std(dim=0)
pca = PCA(n_components=30).fit(n_z_vals)
loadings = pca.transform(n_z_vals)
pca_rand = PCA(n_components=30).fit(torch.distributions.Normal(0, 1).sample(z_vals.shape))

fig, ax = plt.subplots(1, 1)
ax.plot(pca.explained_variance_)
ax.plot(pca_rand.explained_variance_, '--')
fig.show()

# %% Showing some of the plots
fig, axes = plt.subplots(3, 4, figsize=(20, 20))
axes = axes.reshape(-1)
plot_idx = 0
for i in range(5):
    for j in range(i+1, 5):
        axes[plot_idx].scatter(loadings[:, i], loadings[:, j], c=interpolator(densities), alpha=0.1, cmap='inferno')
        axes[plot_idx].set_xlabel(i)
        axes[plot_idx].set_ylabel(j)
        axes[plot_idx].set_xlim(-20, 20)
        axes[plot_idx].set_ylim(-20, 20)
        plot_idx += 1
fig.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.scatter(loadings[:, 0], loadings[:, 1], c=interpolator(densities), alpha=0.1, cmap='inferno')
fig.show()

# %% Identifying observations
# Temperatur bliver for lav:
# 921, 1337, 1749, 1751, 3844, 4688, 5540, 6374, 6803, 8892
chosen = 916
chosen = 8045
chosen = 9328
chosen = 3432
chosen = 6369
chosen = 9731

# High values of the third parameter
chosen = 16597
chosen = 19553
chosen = 18709
chosen = 17862
chosen = 19130
# 11580, 11992, 13246, 14082, 15749, 16172, 17016, 17433, 17853, 19129, 19546

# Temperaturen bliver ikke afkølet nok
chosen = 18085
chosen = 17639
chosen = 10941
chosen = 10397
chosen = 11060


fig, ax = plt.subplots(2, 1, figsize=(7, 7), height_ratios=(3, 1))
ax[0].scatter(loadings[:, 0], loadings[:, 1], c=interpolator(densities), alpha=0.1, cmap='inferno')
ax[0].scatter(loadings[chosen, 0], loadings[chosen, 1], c='red')

x_plot = dataset.X[dataset.list_inds[chosen]][:, 1]
x_window = np.arange(dataset.start_inds[chosen], (dataset.start_inds[chosen]+WINDOW_SIZE))
ax[1].plot(x_plot)
ax[1].plot(x_window, x_plot[dataset.start_inds[chosen]:(dataset.start_inds[chosen]+WINDOW_SIZE)], c='red')
fig.show()

# %%
fig, ax = plt.subplots(1, 1)
ax.scatter(loadings[:, 0], loadings[:, 1])
ax.set_xlim(-10, 0)
# ax.set_xlim(2,4)
ax.set_ylim(-20, -5)
# ax.set_ylim(-8, -6)
fig.show()

np.where(((loadings[:, 0] > -10)*1.0 + (loadings[:, 0] < 0) + (loadings[:, 1] > -20) + (loadings[:, 1] < -5)) == 4)

# %% Identifying some errors:
type1 = []
for i, (l, j) in enumerate(zip(dataset.list_inds, dataset.start_inds)):
    if (dataset.X[l][j:(j+WINDOW_SIZE), 2]).max() > 340:
        type1.append(i)

fig, ax = plt.subplots(1, 1)
ax.scatter(loadings[:, 0], loadings[:, 1], alpha=0.1)
ax.scatter(loadings[type1, 0], loadings[type1, 1], alpha=0.1, c='red')
fig.show()

# %% Making the wildest plot anyone has ever seen
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
        axes[2, 0].set_ylabel('RT temp')
        axes[2, 0].set_xlabel('Time')
fig.savefig('plots/GHL_PCA_clusters_png', bbox_inches='tight', dpi=200)
fig.show()
