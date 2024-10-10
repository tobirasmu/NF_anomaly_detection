from pathlib import Path

class Config:
    def __init__(self):
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


class Dir:
    def __init__(self):
        self.root = Path(__file__).absolute().parent

        self.data = self.root.joinpath('data')
        self.plots = self.root.joinpath('plots')
        self.results = self.root.joinpath('results')
