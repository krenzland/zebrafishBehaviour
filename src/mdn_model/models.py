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
 
class DropoutRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(in_features=input_size,
                            out_features=hidden_size)
        self.h2h = nn.Linear(in_features=hidden_size,
                            out_features=hidden_size,
                            bias=False)
        self.tanh = nn.Tanh()
        self.dropout_p = dropout_p
        
    def forward(self, x, h0=None):
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        # Initialize hidden state
        if h0 is not None:
            hidden = h0
        else:
            hidden = x.new_ones(1, batch_size, self.hidden_size)
        
        # Sample dropout mask for hidden-to-hidden connection.
        if self.training:
            dropout = torch.bernoulli(torch.ones_like(hidden) * self.dropout_p).cuda()
        
        for i in range(seq_len):
            input_res = self.i2h(x[i,:,:])
            if self.training:
                hidden = dropout * hidden
            else:
                hidden = (1-self.dropout_p) * hidden
            hidden_res = self.h2h(hidden)
            hidden = self.tanh(input_res + hidden_res)
         
        # Conform to interface of nn.GRU but only return valid last hidden state.
        return None, hidden

class RNNEncoder(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers=1):
        super().__init__()
               
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        
        # Inital hidden state for one element of batch is learned parameter.
        self.init_hidden = nn.Parameter(torch.randn(self.n_layers,1,n_hidden/self.n_layers))
                
        self.rnn = DropoutRNN(input_size=n_features,
                         hidden_size=n_hidden)
        # For larger sequences you should try using GRU/LSTM!
        # For our purpose they just overfit.
        
        for p in self.parameters():
            if isinstance(p, nn.Linear):
                # Note: Correct initialization is important; otherwise might not converge!
                nn.init.normal_(p, mean=0, std=0.001)
                if p.bias is not None:
                    p.bias.data.fill_(0.0)

    def forward(self, x, batch_size):
        # Expand initial hidden state
        init_hidden = self.init_hidden.expand(self.n_layers, batch_size, self.n_hidden/self.n_layers).contiguous()
        
        _, x = self.rnn(x, init_hidden)
        return x

class StaticSpatialLinearEncoder(nn.Module):
    def __init__(self, n_features, n_dts):
        super().__init__()
        # Treat each timestep as independent feature!
        out_size = 2
        self.n_features = n_features
        # Prior to seeing any data, assume that all timesteps are equally important.
        self.dt_weights = nn.Parameter(data=torch.ones(n_dts)/n_dts)
        
        self.linear = nn.Linear(in_features=self.n_features,
                                out_features=out_size)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, std=0.001) # todo
                m.bias.data.fill_(0.0)
        
    def forward(self, x):
        # Normalize dt_weights s.t. they are >= 0 and sum to 1
        dt_weights = nn.functional.softmax(self.dt_weights, dim=0) 
    
        # Shape: examples, n_dts, output (2)
        x =  self.linear(x)

        x = x.permute(1,2,0)
        return (x * dt_weights).sum(dim=-1)

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
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    m.bias.data.fill_(0)


    def forward(self, hidden, batch_size):
        pi = self.z_pi(hidden).view(batch_size, self.n_components)
        pi = nn.functional.softmax(self.z_pi(hidden), 1)
        
        if self.covariance_type == 'diagonal':
            # sigma: variances of components, need to be > 0
            # 10e-6 added for numerical stability
            sigma = self.z_sigma(hidden).view(batch_size, self.n_components, -1)
            sigma = torch.exp(sigma) + self.covariance_reg
        else:
            sigma_ = self.z_sigma(hidden).view(batch_size, self.n_components, -1)

            # Take sqrt of regularisation, covariance matrix squares these values.
            # sigma0 and sigma2 (diagonals of Cholesky matrix) need to be strictly positive.
            sigma0 = torch.exp(sigma_[:,:,0]) + self.covariance_reg**0.5
            sigma1 = sigma_[:,:,1] # can be arbitrary!
            sigma2 = torch.exp(sigma_[:,:,2]) + self.covariance_reg**0.5
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
        # Remove first dimension of RNN hidden state.
        # Not doing this leads to errors in the NLL-Computation.
        if len(encoded.shape) == 3:
            assert(encoded.shape[0] == 1)
            encoded = encoded.squeeze(0)
        return self.decoder(encoded, batch_size)
