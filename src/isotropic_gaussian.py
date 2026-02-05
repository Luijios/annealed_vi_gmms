import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class MIG(nn.Module):
    def __init__(self, means, vars, weights=None, device='cpu'):
        """
        Class to handle operations around mixtures of multivariate
        Isotropic Gaussian distributions
        Args:
            means: list of 1d tensors of centroids
            vars: list of variances
            weights: list of relative statistical weights (does not need to sum to 1)
        """
        super().__init__()
        self.device = device
    
    # def init_params(self, means, covars, weights=None,):
        self.dim = means[0].shape[0]
        self.k = means.shape[0]  # number of components in the mixture
        self.means = means
        self.vars = vars

        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.tensor([1 / self.k] * self.k, device=device)
        self.cs_distrib = td.categorical.Categorical(probs=self.weights)

    def sample(self, n_batch):
        # return n samples from self
        #output is of shape (n, self.dim)
        cs = self.cs_distrib.sample((n_batch,))#.to(self.device)
        z = torch.randn(n_batch, self.dim, device=self.device)#.double()
        return z * torch.sqrt(self.vars[cs,None]) + self.means[cs]
    
    def nll(self, z):
        #input is of shape (n, self.dim)
        args = - 0.5 * torch.sum( (z[:,None,:] - self.means)**2, dim=-1 ) /  self.vars #args is of size (n, self.k) at this point
        args += torch.log(self.weights) - 0.5*self.dim*torch.log(2*torch.pi*self.vars)
        return - torch.logsumexp(args, dim=-1)

    def renormalize_weights(self, eps=1e-6):
        with torch.no_grad():
            weight_norm = torch.maximum(self.weights.data, torch.ones_like(self.weights) * eps)
            weight_norm = weight_norm / weight_norm.sum()
            self.weights.data = torch.minimum(weight_norm, torch.ones_like(self.weights) - eps)


    def grad_nll(self, x_init):
        x = x_init.detach()
        x = x.requires_grad_()
        optimizer = torch.optim.SGD([x], lr=0)
        optimizer.zero_grad()
        loss = self.U(x).sum()
        loss.backward()
        return x.grad.data

    def log_marginal_density(self, x_range, axis=0):
        # input x_range is of shape (n)
        with torch.no_grad():
            args = - 0.5 * (x_range[:,None] - self.means[:, axis])**2 /  self.vars #args is of size (n, self.k) at this point
            args += torch.log(self.weights) - 0.5*torch.log(2*torch.pi*self.vars)
            return torch.logsumexp(args, dim=-1)