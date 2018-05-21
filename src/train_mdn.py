#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse

import torch
from torch.autograd import Variable
from torch.utils import data
from torch import nn
from torch import optim
from torch.utils import data

from util import add_angles, angle_between, angled_vector, sub_angles
from mdn_model.mixture_loss import *
from mdn_model.models import *

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

def save_checkpoint(model, optimizer, epoch, filename):
    state = {'epoch': epoch + 1,
             'model': model.state_dict(),
             'optim': optimizer.state_dict()}
    torch.save(state, filename)
    

def main():
    # Set up command line arguments.
    parser = argparse.ArgumentParser(description="Run training for NN social model.")
    parser.add_argument('--seed', type=int,
                        help="Value for random seed. Default: Random.")
    parser.add_argument('--checkpoint-name', type=str, required=True,
                        help="Set name for checkpoint.")

    # ---------------------- Model settings ---------------------------------------
    parser.add_argument('--encoder', type=str, default='mlp', choices=['mlp', 'rnn'],
                        help="Set encoder architecture. Default: rnn")
    parser.add_argument('--decoder', type=str, default='mdn', choices=['mdn', 'mse'],
                        help="Set decoder architecture and corresponding loss. Default: mdn")
    parser.add_argument('--hidden-size', type=int, default=64,
                        help="Set size of the hidden units. Default: 64")
    parser.add_argument('--covariance-type', type=str, default='diagonal', choices=['diagonal', 'general'],
                        help="Set covariance type for the mdn decoder. Default: diagonal")
    parser.add_argument('--n-components', type=int, default=5,
                        help="Set number of mixture components for mdn decoder. Default: 5")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Set number of epochs to run. Default: 1000")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help="Set number of epochs to run. Default: 1000")
    parser.add_argument('--lr-decay', type=float, default=0.99,
                        help="Set by how much the lr should be multiplied after each epoch. Default: 0.99")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help="Set weight decay. Default: 1e-4")
    #parser.add_argument('--dropout', type=float, default=0.5,
    #                    help="Set dropout probability for encoder (Hidden-To-Hidden). Default: 0.5")

    args = parser.parse_args()
    print("Called with args={}".format(args))

    # Generate and print random seed:
    if args.seed:
        seed = int(args.seed)
    else:
        seed = np.random.randint(0, 10000)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"Using seed {seed}")
    
    # Set up pytorch device, use cuda if possible.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for training.")

    print("Loading data...")
    # Load data.
    df_train = pd.read_csv( '../data/processed/rf_train.csv')
    df_test = pd.read_csv( '../data/processed/rf_test.csv')

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

    # Set up data loader.
    train_data = LSTMDataset(X_train, y_train)

    train_loader = data.DataLoader(train_data,
                                batch_size=128, 
                                collate_fn=collate,
                                pin_memory=True,
                                num_workers=6,
                                shuffle=True)

    print("Finished data loading.")
    
    # Set up models.
    covariance_type = args.covariance_type

    n_hidden = args.hidden_size
    #encoder = RNNEncoder(n_features=n_features, n_hidden=n_hidden, n_layers=1)

    if args.encoder == 'rnn':
        encoder = RNNEncoder(n_features=n_features, n_hidden=n_hidden)
    else:
        encoder = MLPEncoder(n_features=n_features, n_hidden=n_hidden)
    if args.decoder == 'mdn':
        decoder = MDNDecoder(n_hidden=n_hidden, n_components=args.n_components, covariance_type=covariance_type, covariance_reg=1e-6)
        criterion = MixtureLoss(covariance_type=covariance_type).to(device)
    else:
        decoder = NormalDecoder(n_hidden=n_hidden)
        criterion = nn.MSELoss().to(device) 

    model = ReceptiveFieldNN(encoder=encoder, decoder=decoder).to(device)
    print(model)

    #ptimizer = optim.Adam(params)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Training loop
    for epoch in range(1,args.epochs+1):
        model.train()
        loss_sum = 0.0
        loss_norm = 0
        for batch in train_loader:
            x, y = [b.to(device) for b in batch]

            optimizer.zero_grad()

            out = model(x)
            if isinstance(model.decoder, MDNDecoder):
                pi, sigma, mu = out
                loss = criterion(pi, sigma, mu, y)
            else:
                loss = criterion(out, y)
            loss.backward()

            # Otherwise gradients might become NaN for RNN training.
            nn.utils.clip_grad_value_(model.parameters(), 10)

            optimizer.step()

            loss_sum += loss * x.shape[0]
            loss_norm += x.shape[0]  
        scheduler.step()
        if (epoch % 20) == 0 or epoch == args.epochs:
            model.eval()
            print(f'{epoch}: \nLR = {optimizer.param_groups[0]["lr"]}')
            print(loss_sum.item()/loss_norm)
            if isinstance(model.decoder, MDNDecoder):
                with torch.no_grad():
                    print(criterion(*model(X_train.to(device)), y_train.to(device)), '\n', 
                    criterion(*model(X_test.to(device)), y_test.to(device)))

        # Save checkpoint every few epochs.
        if (epoch % 100) == 0 or epoch == args.epochs:
            print(f'Epoch {epoch}: Saved checkpoint.')
            save_checkpoint(model, optimizer, epoch, f'{args.checkpoint_name}_{epoch}.pt')

if __name__ == '__main__':
    main()
