import csv
import pandas as pd
from tqdm import tqdm
from pathlib import PosixPath, Path
import torch
import pickle
import warnings


def load_pickle(save_path: PosixPath):
    save_path = Path(save_path)
    with open(save_path.with_suffix('.pkl'), 'rb') as file:
        x = pickle.load(file)
    return x


def load_GHL_data(data_dir, saved=True, relabeled=True):
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


def load_SWaT_data(data_dir, saved=True, remove_warmup=False):
    """
    According to the authors of the dataset there is warm-up period of around six hours
    """
    if saved:
        X, L, T, var_names = torch.load(data_dir.joinpath('loaded_data.pt'))
    else:
        # The 'normal' data without any attack
        x_normal = pd.read_excel(data_dir.joinpath('SWaT_Dataset_Normal_v1.xlsx'), header=1)
        t_normal = pd.to_datetime(x_normal[' Timestamp'], format=' %d/%m/%Y %I:%M:%S %p').values
        x_normal = x_normal.iloc[:, 1:(-1)]
        var_names = x_normal.keys()
        x_normal = torch.Tensor(x_normal.values)

        # The 'abnormal' data with attacks
        x_abnormal = pd.read_excel(data_dir.joinpath('SWaT_Dataset_Attack_v0.xlsx'), header=1)
        t_abnormal = pd.to_datetime(x_abnormal[' Timestamp'], format=' %d/%m/%Y %I:%M:%S %p').values
        L_abnormal = torch.Tensor((x_abnormal['Normal/Attack'] != 'Normal').values * 1.0)
        x_abnormal = x_abnormal.iloc[:, 1:(-1)].values
        x_abnormal = torch.Tensor(x_abnormal)

        # Making one list for both
        X = [x_normal, x_abnormal]
        L = [torch.zeros((x_normal.shape[0])), L_abnormal]
        T = [t_normal, t_abnormal]

        torch.save((X, L, T, var_names), data_dir.joinpath('loaded_data.pt'))
    if remove_warmup:
        six_hours = 6 * 60 * 60
        X[0] = X[0][six_hours:]
        L[0] = L[0][six_hours:]
        T[0] = T[0][six_hours:]
    return X, L, T, var_names
