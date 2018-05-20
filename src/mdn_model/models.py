import torch
from torch import nn

class MLPEncoder(nn.Module):
    def __init__(self, n_features, n_hidden):
        super().__init__()
               
        self.in_to_hidden = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Dropout(),           
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=0.001)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, x, batch_size): 
        x = x[-1,:,:] # Ignore all but dt=0
        hidden = self.in_to_hidden(x)
         
        return hidden 
 
class RNNEncoder(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1):
        super().__init__()
               
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        # Inital hidden state for one element of batch is learned parameter.
        self.init_hidden = nn.Parameter(torch.randn(self.n_layers,1,n_hidden/self.n_layers))
        
        # First preprocess with one linear layer.
        preprocess = nn.Linear(in_features=n_features,
                              out_features=n_hidden)
        relu = nn.ReLU(inplace=True)
        dropout = nn.Dropout()
        
        rnn = nn.GRU(input_size=n_hidden,
                                   hidden_size=n_hidden//n_layers,
                                   num_layers=n_layers)
        
        self.in_to_hidden = nn.ModuleList([
            preprocess, relu, dropout, rnn, dropout])
        
        for p in self.parameters():
            if isinstance(p, nn.GRU) or isinstance(p, nn.Linear):
                # Note: Correct initialization is important; otherwise might not converge!
                nn.init.normal_(p, mean=0, std=0.02)

    def forward(self, x, batch_size):
        # Expand initial hidden state
        init_hidden = self.init_hidden.expand(self.n_layers, batch_size, self.n_hidden/self.n_layers).contiguous()
        
        for m in self.in_to_hidden:
            if isinstance(m, nn.GRU):
                # Assumes that we only have one GRU layer!
                _, x = m(x, init_hidden)
                x = x.view(-1, self.n_hidden)
            else:
                # Other preprocessing layers.
                x = m(x)               

        return x

# https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
class MDNDecoder(nn.Module):
    def __init__(self, n_hidden, n_components, covariance_type='diagonal', covariance_reg=1e-6):
        super().__init__()
        
        self.n_components = n_components
               
        n_output = 2        
                
        self.covariance_type = covariance_type
        self.covariance_reg = covariance_reg
        
        if covariance_type == 'diagonal':
            # Predict only a variance per output.
            n_params_sigma = n_components * n_output
        else:
            # Predict parameters of the Cholesky decomposition of the covariance matrix.
            # -> Write cov. matrix sigma = LL^T, with L = [[a, 0], [b, c]]
            assert(n_output == 2) # only supported for 2d
            assert(covariance_type == 'general')
            n_params_sigma = n_components * (n_output + 1)
            
        n_params_mu = n_components * n_output
        
        # Transform hidden layer linearly to parameters of Gaussian mixture.
        self.z_pi = nn.Linear(n_hidden, n_components)
        self.z_sigma = nn.Linear(n_hidden, n_params_sigma)
        self.z_mu = nn.Linear(n_hidden, n_params_mu)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=5.0/3)


    def forward(self, hidden, batch_size):
        pi = self.z_pi(hidden).view(batch_size, self.n_components)
        pi = nn.functional.softmax(self.z_pi(hidden), 1)
        
        if self.covariance_type == 'diagonal':
            # sigma: variances of components, need to be > 0
            # 10e-6 added for numerical stability
            sigma = self.z_sigma(hidden).view(batch_size, self.n_components, -1)
            sigma = nn.functional.softplus(sigma) + self.covariance_reg
        else:
            sigma_ = self.z_sigma(hidden).view(batch_size, self.n_components, -1)

            # Take sqrt of regularisation, covariance matrix squares these values.
            # sigma0 and sigma2 (diagonals of Cholesky matrix) need to be strictly positive.
            sigma0 = nn.functional.softplus(sigma_[:,:,0]) + self.covariance_reg**0.5
            sigma1 = sigma_[:,:,1] # can be arbitrary!
            sigma2 = nn.functional.softplus(sigma_[:,:,2]) + self.covariance_reg**0.5
            sigma = torch.stack((sigma0, sigma1, sigma2), dim=-1)
        
        # mu: mean, can be arbitrary
        mu = self.z_mu(hidden).view(batch_size, self.n_components, -1)         

        return [pi, sigma, mu]
    
    
class NormalDecoder(nn.Module):
    def __init__(self, n_hidden):
        super().__init__()
        # Just a linear transformation of hidden layer.
        self.hidden_to_out = nn.Linear(in_features=n_hidden, out_features=2)
        
    def forward(self, hidden, batch_size):
        return self.hidden_to_out(hidden)
    
class ReceptiveFieldNN(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        batch_size = x.shape[1]
        encoded = self.encoder(x, batch_size)
        return self.decoder(encoded, batch_size)
