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
from data import LSTMDataset, collate, get_data

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
    parser.add_argument('--encoder', type=str, default='mlp', choices=['mlp', 'rnn', 'static'],
                        help="Set encoder architecture. Static corresponds to linear model with static temportal weights. Default: rnn")
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
    X_train, y_train, X_test, y_test = get_data(dir='../data/processed/')
    n_features = X_train.shape[2]
    
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

    if args.encoder == 'static':
        model = StaticSpatialLinearEncoder(n_features=n_features, n_dts=X_train.shape[0]).to(device)
        print("Warning: Using static spatial linear encoder, using no explicit decoder!")
        assert(args.decoder == 'mse')
    else:
        model = ReceptiveFieldNN(encoder=encoder, decoder=decoder).to(device)
    print(model)

    optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.lr, momentum=0.9, nesterov=True)
    # Adam might be a better choice for more complex models and larger data, SGD works well for our model/data.
    #ptimizer = optim.Adam(model.parameters())
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
            if isinstance(criterion, MixtureLoss):
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
            print(f'Train loss over epoch: {loss_sum.item()/loss_norm}')

            if isinstance(criterion, MixtureLoss):
                with torch.no_grad():
                    print('(Eval) Train: ', criterion(*model(X_train.to(device)), y_train.to(device)).item(), '\nTest: ', 
                          criterion(*model(X_test.to(device)), y_test.to(device)).item())
            else:
                with torch.no_grad():
                    print('(Eval) Train: ', criterion(model(X_train.to(device)), y_train.to(device)).item(), '\nTest: ', 
                          criterion(model(X_test.to(device)), y_test.to(device)).item())

        # Save checkpoint every few epochs.
        if (epoch % 100) == 0 or epoch == args.epochs:
            print(f'Epoch {epoch}: Saved checkpoint.')
            save_checkpoint(model, optimizer, epoch, f'{args.checkpoint_name}_{epoch}.pt')

if __name__ == '__main__':
    main()
