from torch.utils.data import Dataset
import torch
from tqdm import tqdm

from config import Config

cfg = Config()

"""
class GHL(Dataset):
    def __init__(self, x, x_a=None, window_size=cfg.model.window_size, stride_size=cfg.model.stride):
        self.x = x
        self.means, self.sds = self.x.mean(dim=0), self.x.std(dim=0)
        self.T, self.D = x.shape
        self.window_size = window_size
        self.stride_size = stride_size

        self.affected = x_a
        self.data, self.idx = self.preprocess(x)

    def preprocess(self, x):
        start_idx = torch.arange(0, self.T - self.window_size, self.stride_size)
        x = (x - self.means) / self.sds
        return x, start_idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        start = self.idx[item]
        end = start + self.window_size
        data = self.data[start:end]

        return torch.moveaxis(data, 0, 1)
"""

class GHL_new(Dataset):
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


class NF_NEW(Dataset):
    def __init__(self, Z, L, MSE, obs_weight=None):
        self.Z = Z
        self.L = L
        self.MSE = MSE
        self.obs_weight = obs_weight if obs_weight is not None else torch.ones(len(self.Z))

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, item):
        return self.Z[item], self.L[item], self.MSE[item], self.obs_weight[item]


class NF_TDC(Dataset):
    def __init__(self, Z, obs_weight=None):
        self.Z = Z
        self.obs_weight = obs_weight if obs_weight is not None else torch.ones(len(self.Z))

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, item):
        return self.Z[item], self.obs_weight[item]


class GHL_NF(Dataset):
    def __init__(self, X, L, model, window_size=120, stride_size=1, offset=0, means=None, sds=None,
                 device=torch.device('cpu')):
        self.window_size = window_size
        self.stride_size = stride_size
        self.X, self.cond, self.lab = self.encode_windows(self.preprocess(X, means, sds), L, model, device, offset=offset)
        self.cond = self.post_process(self.cond)

    def preprocess(self, X, means, sds):
        # Translating
        if means is None:
            temp = []
            for x in X:
                means = x.mean(dim=0)
                temp.append(x - means)
            X = temp
        else:
            X = [x - means for x in X]
        # Scaling
        if sds is None:
            temp = []
            for x in X:
                sds = x.std(dim=0)
                temp.append(x / sds)
            X = temp
        else:
            X = [x / sds for x in X]
        return X

    @torch.no_grad()
    def encode_windows(self, X, L, model, device, chunk_size=400, offset=0):
        model.to(device)
        model.eval()
        # Turning into windows - one file at a time
        values, embeddings, labels = [], [], []
        for l, x in zip(L, X):
            for r in tqdm([range(i, min(i + chunk_size, x.shape[0] - self.window_size - offset), self.stride_size) for i in range(0, x.shape[0] - self.window_size - offset, chunk_size)], desc='Encoding windows'):
                this_val, this_hist, this_lab = [], [], []
                for i in r:
                    this_val.append(x[i + self.window_size + offset])
                    this_hist.append(x[i:(i+self.window_size)])
                    this_lab.append(l[i:(i+self.window_size)])
                this_hist = torch.moveaxis(torch.stack(this_hist), 1, -1).float()
                enc, _ = model(this_hist.to(device))
                values.append(torch.stack(this_val).detach().cpu())
                embeddings.append(enc.detach().cpu())
                labels.append(torch.stack(this_lab).detach().cpu())
        return torch.concatenate(values, dim=0), torch.concatenate(embeddings, dim=0), torch.concatenate(labels, dim=0)

    def post_process(self, cond):
        cond = cond - cond.mean(dim=0)
        cond = cond / cond.std(dim=0)
        return cond

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, item):
        # Returning a value, a historic window, and a label (anomaly or not)
        return self.X[item], self.cond[item], self.lab[item]
