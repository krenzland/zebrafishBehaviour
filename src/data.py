import numpy as np
import pandas as pd
import torch
from torch.utils import data

def get_data(dir):
    # Load data.
    df_train = pd.read_csv( f'{dir}/rf_train.csv')
    df_test = pd.read_csv( f'{dir}/rf_test.csv')

    # Massage data:
    # Put dt column last for both dfs.
    # Otherwise reshaping will have weird results.
    cols = list(df_train.columns)
    cols.remove('dt')
    df_train = df_train[cols + ['dt']]
    df_test = df_test[cols + ['dt']]

    y_labels = ['y_0', 'y_1']
    y_train_np = df_train.loc[df_train['dt'] == 0, y_labels].values
    y_test_np = df_test.loc[df_train['dt'] == 0, y_labels].values

    n_dts = len(np.unique(df_train['dt']))
    n_features = df_train.shape[1] - 3# # minus rows dt, y_0, y_1

    X_train_np = df_train.drop(labels=y_labels + ['dt'], axis=1).values
    X_test_np = df_test.drop(labels=y_labels + ['dt'], axis=1).values

    # Reshape from (dts*kicks, features) to (kicks, dts, n_features)
    X_train_np = X_train_np.reshape(X_train_np.shape[0]//n_dts, n_dts, n_features)
    X_test_np = X_test_np.reshape(X_test_np.shape[0]//n_dts, n_dts, n_features)

    # Transpose to (dts, kicks, features)
    X_train_np = X_train_np.transpose(1,0, 2)
    X_test_np = X_test_np.transpose(1,0, 2)

    # Reverse order of timesteps s.t. dt=0 is last
    X_train_np = X_train_np[::-1,:,:].copy()
    X_test_np = X_test_np[::-1,:,:].copy()

    # To Tensors
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()

    X_test = torch.from_numpy(X_test_np).float()
    y_test = torch.from_numpy(y_test_np).float()

    return X_train, y_train, X_test, y_test
    
class LSTMDataset(data.Dataset):
    def __init__(self, X, y):
        super(data.Dataset, self).__init__
        
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        return self.X[:, idx,:], self.y[idx,:]

def collate(batch):
    result = []
    Xs = []
    ys = []
    for elem in batch:
        X = elem[0]
        y = elem[1]
        Xs.append(X)
        ys.append(y)
    return torch.stack(Xs, dim=1), torch.stack(ys, dim=0)

