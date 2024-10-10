import pandas as pd
from tqdm import tqdm
from pathlib import Path, PosixPath
import torch
import pickle
import warnings


def load_pickle(save_path: PosixPath):
    save_path = Path(save_path)
    with open(save_path.with_suffix('.pkl'), 'rb') as file:
        x = pickle.load(file)
    return x


def load_GHL_data(data_dir, saved=True, relabeled=True):
    """
    Loading the GHL dataset. Notice that if 'saved' is set to False one needs to first download the GHL dataset from
    "https://kas.pr/ics-research/dataset_ghl_1" and unzip the files into the GHL folder.

    :param data_dir: Path to the GHL folder
    :param saved: Boolean stating if the saved version should be used.
    :param relabeled: Boolean stating if the relabeled version should be used.
    :return: The time series, labels, and time points all given as torch.Tensor
    """
    if saved:
        if relabeled:
            X, L, T = torch.load(data_dir.joinpath('relabeled_loaded_data.pt'))
        else:
            X, L, T = torch.load(data_dir.joinpath('loaded_data.pt'))
    else:
        if relabeled:
            warnings.warn('Not loading the relabeled version -- Relabeled dataset is only accessible in the saved version.')
        # The training data
        x_train = pd.read_csv(data_dir.joinpath('train_1500000_seed_11_vars_23.csv'))
        t_train = x_train.Time.values
        x_train = x_train[['RT_level', 'RT_temperature.T', 'HT_temperature.T', 'inj_valve_act', 'heater_act']]
        var_names = list(x_train.keys())
        x_train = torch.Tensor(x_train.values)

        # The testing data
        X, T, L = [], [], []
        for file in tqdm(sorted(data_dir.glob('*_Temp_*_vars_23.csv')), total=48,
                         desc='Loading anomalous data'):
            x = pd.read_csv(file)
            T.append(torch.Tensor(x.Time.values))
            L.append(torch.Tensor(x['DANGER'].values))
            x = x[['RT_level', 'RT_temperature.T', 'HT_temperature.T', 'inj_valve_act', 'heater_act']]
            # V_names.append(list(x.keys()))
            X.append(torch.Tensor(x.values))

        # Making one list for both
        X.append(x_train)
        L.append(torch.zeros((x_train.shape[0])))
        T.append(torch.tensor(t_train))

        torch.save((X, L, T), data_dir.joinpath('loaded_data.pt'))
    return X, L, T
