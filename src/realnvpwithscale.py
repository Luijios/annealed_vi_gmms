__author__ = "Luigi Fogliani"
__date__ = "2026/02/10"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional
from torch.distributions import Distribution, MultivariateNormal



###############################
#### AFFINE COUPLING LAYER ####
###############################

class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fct = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            nn.Tanh() #for a mysterious reason, using ReLU makes the training impossible, when scale and translation have same depth
        )

        ## layers used to compute translation in affine transformation
        self.translation_fct = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim),
            #nn.Tanh()
        )

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        return self.scale_fct(self.mask*x) #* self.scale

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network
        return self.translation_fct(self.mask*x)
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)
        logdet = torch.sum((1 - self.mask)*s, -1)
        #y = self.mask*x + (1-self.mask)*(x*2*torch.sigmoid(s) + t)
        #logdet = torch.sum((1 - self.mask)*torch.log(2*torch.sigmoid(s)), -1)
        
        return y, logdet

    def inverse(self, y):
        ## convert observed variable to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)
                
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        #x = self.mask*y + (1-self.mask)*((y - t)/2/torch.sigmoid(s))
        #logdet = torch.sum(-(1 - self.mask)*torch.log(2*torch.sigmoid(s)), -1)
        
        return x, logdet




##################
#### REAL NVP ####
##################
    
class RealNVP(nn.Module):
    '''
    A vanilla RealNVP class with alternating checkerboard pattern masks
    '''
    
    def __init__(self, num_layers, dim, hidden_dim, prior=None, device='cpu', use_scale_and_shift=False, use_batch_norm_between_layers=False):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RealNVP, self).__init__()
        self.device=device
        if prior is None:
            self.prior = MultivariateNormal(torch.zeros(dim, device=device), torch.eye(dim, device=device))
        else:
            self.prior = prior
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.num_layers = num_layers

        self.masks = torch.zeros(num_layers, dim)
        self.masks[::2, ::2] = 1.0
        self.masks[1::2, 1::2] = 1.0

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim).to(device) for i in range(self.num_layers)]
        )

        # if self.use_scale_and_shift:
        #     self.affine_couplings.append(
        #         scale_and_shift(self.dim, momentum=0.5).to(device)
        #     )
        #     self.num_layers += 1

        
    def forward(self, z):
        ## convert latent space variables into observed variables
        x = z.clone()
        logdet_tot = torch.zeros(x.shape[0], device=self.device)
        for i in range(self.num_layers):
            x, logdet = self.affine_couplings[i](x)
            logdet_tot = logdet_tot + logdet
        return x, logdet_tot

    def inverse(self, x):
        ## convert observed variables into latent space variables        
        z = x.clone()    
        logdet_tot = torch.zeros(z.shape[0], device=self.device)
        ## inverse affine coupling layers
        for i in range(self.num_layers-1, -1, -1):
            z, logdet = self.affine_couplings[i].inverse(z)
            logdet_tot = logdet_tot + logdet
            
        return z, logdet_tot

    

### SCALE AND SHIFT ### Not used in the end

class scale_and_shift(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.logweight = nn.Parameter(torch.zeros(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(0)
                batch_var = x.var(0, unbiased = False)
                self.running_mean.mul_(1 - self.momentum).add_(batch_mean * self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(batch_var * self.momentum)
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        std = torch.sqrt(var + self.eps) # Standard deviation (add eps for stability)

        x_hat = (x - mean) / std # Normalize: (x - mean) / std
        y = x_hat * torch.exp(self.logweight) + self.bias # Apply affine transformation:
        # Log-determinant calculation
        log_det = torch.sum(self.logweight - torch.log(std), -1) # Jacobian is diagonal: diag(weight / std)

        return y, log_det
    
    def inverse(self, y):
        mean = self.running_mean
        var = self.running_var
        std = torch.sqrt(var + self.eps)

        x_hat = (y - self.bias) / torch.exp(self.logweight) # Reverse affine: (y - beta) / gamma
        x = x_hat * std + mean # Reverse normalization: x_hat * std + mean
        # Inverse log_det is simply the negative of the forward log_det
        log_det = -torch.sum(self.logweight - torch.log(std), -1)

        return x, log_det

