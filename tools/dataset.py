from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from config import Config

cfg = Config()


class GHL(Dataset):
    def __init__(self, X, L, window_size=cfg.model.window_size, stride_size=cfg.model.window_size,
                 list_indices=None, start_indices=None, means=None, stds=None, obs_weights=None):
        self.X = X
        self.L = L
        self.window_size = window_size
        self.stride = stride_size

        # If the indices are not given
        if list_indices is None or start_indices is None:
            self.list_inds, self.start_inds = self.make_obs()
        else:
            self.list_inds, self.start_inds = list_indices, start_indices
        if means is None or stds is None:
            self.means, self.stds = self.make_stats()
        else:
            self.means = means
            self.stds = stds

        # weighted observations
        if obs_weights is None:
            self.obs_weights = torch.ones(len(self.list_inds))
        elif len(obs_weights) != len(self.list_inds):
            raise ValueError(f'Weights should be a vector of length {len(self.list_inds)}!')
        else:
            if isinstance(obs_weights, torch.Tensor):
                self.obs_weights = obs_weights.clone().detach()
            else:
                self.obs_weights = torch.tensor(obs_weights)

    def make_obs(self):
        lengths = [x.shape[0] for x in self.X]
        num_windows_list = [(L - self.window_size) // self.stride + 1 for L in lengths]
        list_inds = [index for index, i in enumerate(num_windows_list) for _ in range(i)]
        start_inds = [i for L in lengths for i in range(0, L - self.window_size + 1, self.stride)]
        return list_inds, start_inds

    def make_stats(self):
        means, vars = torch.zeros((self.X[0].shape[1], )), torch.zeros((self.X[0].shape[1], ))
        N = 0
        for x in self.X:
            n = len(x)
            N += n
            means += x.mean(dim=0) * n
            vars += x.var(dim=0) * (n-1)

        means /= N
        vars /= (N - 1)
        # If the variable was / is binary we dont standardize. We assume that checking one is enough
        binary = torch.where(torch.logical_and(self.X[-1] != 0.0, self.X[-1] != 1.0).sum(dim=0) == 0)[0]
        means[binary] = 0.0
        vars[binary] = 1.0
        return means, torch.sqrt(vars)

    def __len__(self):
        return len(self.list_inds)

    def __getitem__(self, item):
        list_ind = self.list_inds[item]
        start_ind = self.start_inds[item]
        end_ind = start_ind + self.window_size
        return (((self.X[list_ind][start_ind:end_ind] - self.means) / self.stds).T,
                self.L[list_ind][start_ind:end_ind],
                self.obs_weights[item])


class GHL_NF(Dataset):
    def __init__(self, Z, L, MSE, obs_weight=None):
        self.Z = Z
        self.L = L
        self.MSE = MSE
        self.obs_weight = obs_weight if obs_weight is not None else torch.ones(len(self.Z))

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, item):
        return self.Z[item], self.L[item], self.MSE[item], self.obs_weight[item]
