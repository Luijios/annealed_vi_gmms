import torch
import numpy as np
import copy
import scipy.integrate
from scipy.special import expit


##### Theory #####

def f(s, w_1, R, u=1):
    return scipy.integrate.quad(lambda x: (w_1 * expit(R**2 / u**2 * (s - 1) + R / u * np.sqrt(2*(1 - s)) * x + np.log((1-w_1)/w_1))**2 + 
                                (1-w_1) * expit(R**2 / u**2 * (s - 1) + R / u * np.sqrt(2*(1 - s)) * x + np.log(w_1/(1-w_1)))**2) * np.exp(-x**2 / 2) / (2 * np.pi)**(1/2), -np.inf, np.inf)[0]
def g(m, w_star, R, u=1):
    return 1 - 2 * scipy.integrate.quad(lambda x: expit(2 * R**2 * m + 2 * R * u * x + np.log(w_star/(1-w_star))) * np.exp(-x**2 / 2) / (2*np.pi)**(1/2), -np.inf, np.inf)[0]

def g_prime(m, w_star, R, u=1, eps=1e-5):
    return (g(m+eps, w_star, R, u) - g(m-eps, w_star, R, u))/2/eps

##### ANNEALING SCHEDULES #####

def annealing_schedule(t, param):
    t0 = param['t0']
    beta_i = param['beta_i']
    type = param['type']

    if type=='step': return np.where(t<t0, beta_i, 1)
    if type=='exp': return np.where(beta_i**(1-t/t0)<1, beta_i**(1-t/t0), 1)
    if type is None : return np.ones(t)

def heating_exp(t,t0,beta_i) : return np.where(beta_i**(1-t/t0)<1, beta_i**(1-t/t0), 1)
def heating_step(t,t0,beta_i) : return np.where(t<t0, beta_i, 1)


def log_marginal_along_mu(R, m, sigma2, weights, x_range):
    means = R*m
    args = - 0.5 * (x_range[:,None] - means)**2 /  sigma2 #args is of size (n, 2) at this point
    args += torch.log(weights) - 0.5*torch.log(2*torch.pi*sigma2)
    return torch.logsumexp(args, dim=-1)