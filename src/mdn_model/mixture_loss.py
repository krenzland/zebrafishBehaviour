import numpy as np
import torch
from torch import nn

def determinant(L):
    """Computes determinant of LL^T, assuming that L is triagonal matrix."""
    return (L[:,:,0] * L[:,:,2])**2

def solve(L, b):
    """Solves linear system (LL^T)x = b for lower-diagonal matrices L."""
    def forward(L, b):
        # This avoids in-place modifications of pytorch variables and allows
        # the computation of a gradient.
        x0 = b[:,:,0] / L[:,:,0]
        x1 = (b[:,:,1] - L[:,:,1] * x0)/L[:,:,2]
        return torch.stack((x0, x1), dim=-1)

    def backward(L_t, b):
        x1 = b[:,:,1] / L_t[:,:,2]
        x0 = (b[:,:,0] - L_t[:,:,1] * x1)/L_t[:,:,0]
        return torch.stack((x0, x1), dim=-1)

    # Use forward-backward substitution: efficient and numerically stable.
    return backward(L, forward(L, b))

def multivariate_gaussian_log(x, mu, sigma):
    """
    Compute log pdf of multivariate gaussian with general covariance sigma.
    X: (N, d)
    mu: (N, K, d)
    sigma: (N, K, 3) for d=2, where last dimension contains parameters of Cholesky decomposition.
    return: pdf (K, N, d)
    """
    norm_factor = ((2*np.pi)**2 * determinant(sigma)) ** -0.5
    K = mu.size(1)
    x = x.unsqueeze(1).repeat(1,K,1) # N, K, d
    dist = x - mu

    # Inverting the covariance matrix here is unstable.
    # Thus solve linear system instead.
    exponent = -0.5 * torch.einsum(
        'abij,abji->abi',
        (dist.unsqueeze(-2), solve(sigma, dist).unsqueeze(-1))).squeeze(-1) # N, K
    pdf = exponent + norm_factor.log()
   
    return pdf

def multivariate_gaussian_diagonal_log(x, mu, sigma):
    """
    Compute log pdf of multivariate gaussian with diagonal covariance sigma.
    X: (N, d)
    mu: (N, K, d)
    sigma: (N, K, d)
    return: pdf (K, N, d)
    """
    covar_inv = 1 / sigma # N, K, d
    norm_factor = (2 * np.pi * sigma).prod(dim=2) ** -0.5 # N, K
    K = mu.size(1)
    x = x.unsqueeze(1).repeat(1,K,1) # N, K, d
    dist = x - mu
    exponent = -0.5 * (dist**2*covar_inv).sum(dim=-1)  # N, K

    pdf = exponent + norm_factor.log()
    return pdf # N, K    

def log_sum_exp(x, dim):
    """Compute x.exp().sum().log() with the log-sum-exp trick.    
    More robust against underflow."""
    m = x.max() # Forces minimal value to be 0 log(exp(0)) = 0
    return m + (x - m).exp().sum(dim=dim).log()

class MixtureLoss(nn.Module):
    def __init__(self, covariance_type='diagonal', reduce=True):
        super().__init__()
        assert(covariance_type in ['diagonal', 'general'])
        self.covariance_type = covariance_type
        self.reduce = reduce
        
    def forward(self, pi, sigma, mu, y):
        # Compute log likelihood using the log-sum-exp trick
        if self.covariance_type == 'diagonal':
            ll = (multivariate_gaussian_diagonal_log(y, mu, sigma)) + pi.log()
        else:
            ll = (multivariate_gaussian_log(y, mu, sigma)) + pi.log()
        ll = -log_sum_exp(ll, dim=1)
        if self.reduce:
            return ll.mean()
        return ll
