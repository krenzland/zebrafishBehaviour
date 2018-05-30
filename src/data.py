import numpy as np
import pandas as pd
import torch
from torch.utils import data

def process_df(df):
    # Massage data:
    # Put dt column last for both dfs.
    # Otherwise reshaping will have weird results.
    cols = list(df.columns)
    cols.remove('dt')
    df = df[cols + ['dt']]

    y_labels = ['y_0', 'y_1']
    y_np = df.loc[df['dt'] == 0, y_labels].values

    n_dts = len(np.unique(df['dt']))
    n_features = df.shape[1] - 3# # minus rows dt, y_0, y_1

    X_np = df.drop(labels=y_labels + ['dt'], axis=1).values

    # Reshape from (dts*kicks, features) to (kicks, dts, n_features)
    X_np = X_np.reshape(X_np.shape[0]//n_dts, n_dts, n_features)

    # Transpose to (dts, kicks, features)
    X_np = X_np.transpose(1,0, 2)

    # Reverse order of timesteps s.t. dt=0 is last
    X_np = X_np[::-1,:,:].copy()

    # To Tensors
    X = torch.from_numpy(X_np).float()
    y = torch.from_numpy(y_np).float()

    return X, y

def get_data(dir):
    # Load data.
    df_train = pd.read_csv( f'{dir}/rf_train.csv')
    df_test = pd.read_csv( f'{dir}/rf_test.csv')

    X_train, y_train = process_df(df_train)
    X_test, y_test = process_df(df_test)

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

