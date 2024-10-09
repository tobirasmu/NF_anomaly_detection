from pathlib import Path

class Config:
    def __init__(self):
        self.sim_params = SimParams()
        self.dir = Dir()
        self.model = Model()


class Model:
    def __init__(self):
        self.batch_size = 32
        self.test_size = 0.15
        self.val_size = 0.15
        self.train_size = 1.0 - self.test_size - self.val_size

        self.stride = 1
        self.window_size = 64
        self.seed = 42
        self.epochs = 10


class SimParams:
    def __init__(self):
        self.time_series_length = 10000
        self.mean_lin_anomalies = 1000
        self.mean_var_anomalies = 0
        self.mean_spike_anomalies = 0
        self.tree_height = 3
        self.num_modems = 10
        self.internal_degree = 3
        self.phi_1 = 0.4
        self.phi_2 = - 0.2
        self.sigma = 0.1
        self.sigma_add = 0.3


class Dir:
    def __init__(self):
        self.root = Path(__file__).absolute().parent

        self.data = self.root.joinpath('data')
        self.plots = self.root.joinpath('plots')