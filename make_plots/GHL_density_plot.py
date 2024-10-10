import torch
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from tools.dataset import GHL_new
from models.AE import auto_encoder
from tools.reader import load_GHL_data
from models.NF import MAF
import config

cfg = config.Config()

WINDOW_SIZE = 2400
EMB_DIM = 30
iteration = 0

mpl.rc('font', family="Times New Roman")  # Ten")
plt.rcParams['font.size'] = 15
plt.rcParams['mathtext.fontset'] = 'cm'
figwidth = 5.765

model_folder = cfg.dir.results.joinpath(f'density_example')
X, L, _ = load_GHL_data(cfg.dir.data.joinpath('GHL'), saved=True, relabeled=True)
dataset = GHL_new(X, L, window_size=WINDOW_SIZE, stride_size=WINDOW_SIZE // 5, obs_weights=None)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

# %% Plotting the TS including the estimated p-values for each window.
plotting_stride = 100  # Stride size to use for the sliding window approach
plotting_I = -17  # Which modem to use for the example

plotting_data = GHL_new([X[plotting_I]], [L[plotting_I]], window_size=WINDOW_SIZE, stride_size=plotting_stride, means=dataset.means,
                        stds=dataset.stds)
plot_loader = torch.utils.data.DataLoader(plotting_data, batch_size=128, shuffle=False)

autoencoder_model = autoencoder = auto_encoder(5, kernel_size=5, channels=256, emb_dim=EMB_DIM,
                                               window_size=WINDOW_SIZE, variational=False)
autoencoder_model.load_state_dict(torch.load(model_folder.joinpath('AE_model.pt'), map_location='cpu'))
autoencoder_model.eval()

normalizing_flow = MAF(n_blocks=32, input_size=EMB_DIM, hidden_size=256, n_hidden=1, cond_label_size=None,
                       batch_norm=False, activation='tanh')
normalizing_flow.load_state_dict(torch.load(model_folder.joinpath('nf_model.pt'), map_location='cpu'))
normalizing_flow.eval()

Z = []
with torch.no_grad():
    for batch, l, _ in tqdm(plot_loader):
        enc, _ = autoencoder_model(batch)
        Z.append(normalizing_flow.log_prob(enc))
Z = torch.concatenate(Z)


# %% Estimating the density-CDF to obtain the p-value predictor. Notice that number of samples should be chosen large
# for accurate representation.
sampled_densities = []
with torch.no_grad():
    for i in tqdm(range(200)):
        x_rand = torch.randn((1000, EMB_DIM))
        _, jacobian = normalizing_flow.inverse(x_rand)
        sampled_densities.append((torch.distributions.Normal(0, 1).log_prob(x_rand) - jacobian).sum(axis=1).detach().cpu())

estimated_density = torch.concatenate(
    [Z.min()[None], torch.concatenate(sampled_densities), Z.max()[None]])
estimated_density = torch.sort(estimated_density).values
estimated_probabilities = torch.linspace(1e-10, 1, 200002)
interpolator = interp1d(estimated_density, estimated_probabilities)


# %% Plotting
import numpy as np
color_data = torch.concatenate((Z[0].repeat((WINDOW_SIZE + plotting_stride) // 2),
                                Z[1:-1, None].repeat((1, 100)).reshape(-1),
                                Z[-1].repeat(X[plotting_I].shape[0] - (len(Z) - 2)*100 - (WINDOW_SIZE + plotting_stride) // 2)))
normalizer = Normalize(color_data.min(), color_data.max())
cmap = plt.get_cmap('inferno_r')
fig, axes = plt.subplots(4, 1, figsize=(5.765*2, 5.765))
axes = axes.reshape(-1)
variable_names = ['RT Level', 'RT Temp.', 'HT Temp.', 'Est. P-val.']
for i in range(3):
    axes[i].plot(X[plotting_I][:120000, i].numpy())
    axes[i].set_ylabel(variable_names[i])
axes[-1].fill_between(np.arange(120000), np.zeros(120000), L[plotting_I][:120000].numpy(), label='Labeled anom.', color='C1')
axes[-1].set_ylabel(variable_names[-1])
axes[-1].plot(interpolator(color_data)[:120000], label='Mean p-value', color='C2')
axes[-1].plot([0, 120000], [0.01, 0.01], color='red', linestyle='dashed', label='1% Threshold')
axes[-1].set_yscale('log')
axes[-1].set_xlabel('Time')
axes[-1].legend(loc='lower right')
for ax in axes:
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
for ax in axes[:-1]:
    ax.get_xaxis().set_visible(False)
fig.savefig('plots/density_example_GHL.png', dpi=400, bbox_inches='tight')
fig.show()
