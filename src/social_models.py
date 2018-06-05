# TODO: Sort out imports!
from abc import ABC, abstractmethod

import dill as pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import torch
from torch.autograd import Variable
from torch.utils import data
from torch import nn
from torch import optim
from torch.utils import data

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import RidgeCV

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

from util import add_angles, angle_between, angled_vector, sub_angles
from mdn_model.mixture_loss import *
from train_linear import convert_data_concatenate, convert_data_no_memory
from mdn_model.models import *

from data import get_data
from scipy import stats
from matplotlib import colors
from mdn_model.mixture_loss import log_sum_exp

def compute_covar_matrix(sigma):
    sigma = np.vstack([sigma[:,0], np.zeros_like(sigma[:,0]), sigma[:,1], sigma[:,2]]).T
    sigma = sigma.reshape(-1,2,2)
    sigma = np.matmul(sigma,sigma.transpose(0,2,1))
    return sigma

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

def mixture_sample(pi, mu, sigma, n_samples):   
    # First sample from component distribution.
    components = gumbel_sample(np.tile(pi,n_samples).reshape(n_samples,-1))

    components, component_samples = np.unique(components, return_counts=True)

    samples = []
    for idx, size in zip(components, component_samples):
        # Note that without increased tolerance numpy thinks our matrices aren't positive
        # semidefinit even though they are by construction!
        samples.append(np.random.multivariate_normal(mean=mu[idx],
                                                     cov=sigma[idx],
                                                     size=size,
                                                     check_valid='raise',
                                                     tol=1e-6))
    return np.vstack(samples)

def contour_gmm(pi, sigma, mu, ax):
    # Only support one sample per plot.
    sns.set_style('white')
    x = np.linspace(-10,10, num=200)
    y = np.linspace(-10,10, num=200)
    X, Y = np.meshgrid(x,y)
    XX = np.array([X.ravel(), Y.ravel()]).T

    #Z = np.zeros_like(X)*1.0
    Z = []
    mean = np.array([0.,0])
    for p, s, m in zip(pi, sigma, mu):
        gaussian = stats.multivariate_normal(mean=m, cov=(s))
        #print(p.shape, m.shape)
        mean += p*m
        ll = gaussian.logpdf(XX)
        ll = ll.reshape(X.shape)
        Z.append(np.log(p) + ll)
        #Z +=  p * ll

    Z = -log_sum_exp(torch.tensor(Z), dim=0)

    vmin, vmax = Z.min(), Z.max()
    cs = ax.contour(X,Y,Z, levels=np.logspace(0, 2, num=15), norm=colors.SymLogNorm(linthresh=0.01, vmin=vmin, vmax=vmax),)
    #fig.colorbar(cs, shrink=0.8, extend='both')
    #mu_x, mu_y = np.hsplit(mu, [1])
    #ax.scatter(mu_x, mu_y, c='green', marker='x', s=200, zorder=10, linewidth=4, label='component')
    #ax.scatter(vector_gt[0], vector_gt[1], c='blue', marker='x', s=200, zorder=10, linewidth=4, label='gt')
    ax.scatter(mean[0], mean[1], c='red', marker='x', s=200, zorder=10, linewidth=4, label='pred')

    #ax.legend(frameon=True)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

def multivariate_sample(mu_, sigma_):
    return np.vstack([np.random.multivariate_normal(m, s) for m, s in zip(mu_, sigma_)])


def compute_covar_matrix(sigma):
    sigma = np.vstack([sigma[:,0], np.zeros_like(sigma[:,0]), sigma[:,1], sigma[:,2]]).T
    sigma = sigma.reshape(-1,2,2)
    sigma = np.matmul(sigma,sigma.transpose(0,2,1))
    return sigma

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)

def mixture_sample(pi, mu, sigma, n_samples):    
    # First sample from component distribution.
    components = gumbel_sample(np.tile(pi,n_samples).reshape(n_samples,-1))

    components, component_samples = np.unique(components, return_counts=True)

    samples = []
    for idx, size in zip(components, component_samples):
        # Note that without increased tolerance numpy thinks our matrices aren't positive
        # semidefinit even though they are by construction!
        samples.append(np.random.multivariate_normal(mean=mu[idx],
                                                     cov=sigma[idx],
                                                     size=size,
                                                     check_valid='raise',
                                                     tol=1e-6))
    return np.vstack(samples)


class SocialModel(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def predict(self, receptive_field):
        pass
    
    @abstractmethod
    def sample(self, receptive_field, n_samples=1):
        pass
    
    @abstractmethod
    def contour(self, receptive_field, ax):
        pass
    
    @abstractmethod
    def get_required_timesteps(self):
        pass
    
class MixtureModel(SocialModel):
    def __init__(self, model):
        self.model = model
        assert(isinstance(self.model.decoder, MDNDecoder))
        self.covariance_type = self.model.decoder.covariance_type

        if isinstance(self.model.encoder, RNNEncoder):
            self.encoder_type = 'rnn'
        else:
            self.encoder_type = 'mlp'
            
    def evaluate_(self, receptive_field):
        self.model.eval()
        
        # First evaluate the model
        with torch.no_grad():
            out = self.model(receptive_field)
            
        to_numpy = lambda o: o.squeeze(0).detach().cpu().numpy()
        pi, sigma, mu = [to_numpy(o) for o in out]
        
        # Compute covariance matrices
        if self.covariance_type == 'diagonal':
            sigma = np.stack((np.diag(s) for s in sigma), axis=0)
        else:
            sigma = compute_covar_matrix(sigma)
        
        return pi, sigma, mu
        
    def predict(self, receptive_field):
        pi, _, mu = self.evaluate_(receptive_field)
        return (np.stack((pi, pi), axis=-1) *  mu).sum(axis=0)
        
    def sample(self, receptive_field, n_samples=1):
        pi, sigma, mu = self.evaluate_(receptive_field)
            
        samples = mixture_sample(pi, mu, sigma, n_samples=n_samples)
        return samples    

    def contour(self, receptive_field, ax):
        pi, sigma, mu = self.evaluate_(receptive_field)
        #print(pi, '\n', sigma, '\n', mu)
        return contour_gmm(pi, sigma, mu, ax)
    
    def get_required_timesteps(self):
        if self.encoder_type == 'rnn':
            return np.linspace(0, 35, num=35//5+1)
        else:
            return np.array([0])        
        
class MSEModel(SocialModel):
    def __init__(self, model, train_variance, required_timesteps, model_type=None):
        self.model = model
        self.train_variance = train_variance
        self.required_timesteps = required_timesteps

        if isinstance(self.model, nn.Module):
            assert(isinstance(self.model, StaticSpatialLinearEncoder) or
                   isinstance(self.model.decoder, NormalDecoder))
        
            self.model_type = 'torch'
            
        else:
            self.model_type = model_type
            
    def evaluate_torch_(self, receptive_field):
        self.model.eval()
        
        # First evaluate the model
        with torch.no_grad():
            out = self.model(receptive_field).detach().cpu().numpy()
        
        return out.reshape(-1)
    
    def evaluate_sklearn_(self, receptive_field):
        receptive_field = receptive_field.detach().cpu().numpy()
        if 'no_memory' in self.model_type:
            print('no_memory')
            receptive_field = convert_data_no_memory(receptive_field)
        elif 'conc' in self.model_type:
            receptive_field = convert_data_concatenate(receptive_field)
        
        print(receptive_field.shape)
        out = self.model.predict(receptive_field)
        return out.reshape(-1)
        
    def evaluate_(self, receptive_field):
        if isinstance(self.model, nn.Module):
            return self.evaluate_torch_(receptive_field)
        else:
            return self.evaluate_sklearn_(receptive_field)

        
    def predict(self, receptive_field):
        return self.evaluate_(receptive_field)
        
    def sample(self, receptive_field, n_samples=1):
        # TODO: Actually sample, right now just return mean pred
        mu = self.evaluate_(receptive_field)
        return np.random.multivariate_normal(mean=mu,
                                      cov=np.diag(self.train_variance),
                                     size=n_samples)

    def contour(self, receptive_field, ax):
        mu = [self.evaluate_(receptive_field)]
        sigma = [self.train_variance]
        pi = [np.array([1])]
        return contour_gmm(pi, sigma, mu, ax)
    
    def get_required_timesteps(self):
        return self.required_timesteps  

def load_social_model(checkpoint_path, X_train, y_train):
    # When loading a torch checkpoint with pickle we get the following
    # magic number.
    torch_magic_number = 119547037146038801333356
    
    with open(checkpoint_path, 'rb') as file:
        model = pickle.load(file)
    
    print(type(model))
    
    # Order of timesteps gets reversed in data.process_df!
    timesteps_map = {'no_memory': np.array([0]),
                     'memory': np.linspace(start=0, stop=40, num=35//5+1)}
    print(timesteps_map)
    
    data_converter = lambda x: x # convert X_train to appropiate format
    
    prediction_type = None # mean or mdn
    
    if isinstance(model, RidgeCV):
        prediction_type = 'mean'
        if 'conc' in checkpoint_path:
            model_type = 'linear_conc'
            data_converter = lambda x: convert_data_concatenate(x.cpu().numpy())
            required_timesteps = timesteps_map['memory']
        else:
            assert('no_memory' in checkpoint_path)
            model_type = 'linear_no_memory'
            data_converter = lambda x: convert_data_no_memory(x.cpu().numpy())
            required_timesteps = timesteps_map['no_memory']
            
    if model == torch_magic_number:
        n_features = X_train.shape[-1]
        n_dts = X_train.shape[0]
        
        # Other settings are hardcoded.
        n_hidden = 64
        n_components = 5
        covariance_type = 'diagonal'

        # Reload checkpoint, this time correctly with PyTorch.
        checkpoint = torch.load(checkpoint_path)['model']
        
        model_type = 'torch'
        
        # Initialise models
        if 'static' in checkpoint_path:
            model = StaticSpatialLinearEncoder(n_features=n_features,
                                               n_dts=n_dts).to(device)
            prediction_type = ',eam'
            required_timesteps = timesteps_map['memory']
        else:
            # Other models are composed of encoder + decoder
            
            # Set up encoder
            if 'rnn' in checkpoint_path:
                encoder = RNNEncoder(n_features=n_features,
                                    n_hidden=n_hidden)
                required_timesteps = timesteps_map['memory']
            else:
                assert('mlp' in checkpoint_path)
                encoder = MLPEncoder(n_features=n_features,
                                    n_hidden=n_hidden)
                required_timesteps = timesteps_map['no_memory']

                
            # Set up decoder
            if 'mse' in checkpoint_path:
                decoder = NormalDecoder(n_hidden=n_hidden)
                prediction_type = 'mean'
            else:
                assert('mdn' in checkpoint_path)
                decoder = MDNDecoder(n_hidden=n_hidden,
                                    n_components=n_components,
                                    covariance_type=covariance_type)
                prediction_type = 'mdn'

            # Set up final model  
            model = ReceptiveFieldNN(encoder=encoder,
                                    decoder=decoder).to(device)
            model.eval()
                
        # Load checkpoint
        model.load_state_dict(checkpoint)
    
    if 'mean' == prediction_type:
           
        if isinstance(model, RidgeCV):
            y_hat = model.predict(data_converter(X_train))
        else:
            assert(isinstance(model, nn.Module))
            with torch.no_grad():
                y_hat = model(X_train).detach().cpu().numpy()
            
        train_var = np.var(a=(y_train.cpu().numpy() - y_hat),
                           axis=0)
        
        social_model = MSEModel(model=model,
                                train_variance=train_var,
                                model_type=model_type,
                                required_timesteps=required_timesteps)
    else:
        social_model = MixtureModel(model=model)
    
    return social_model
