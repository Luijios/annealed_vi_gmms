import torch
import numpy as np
import json
from src.isotropic_gaussian import *
from src.utils import *

############################
###### TRAINING LOOP #######
############################
    
def train_bimodal(model, target, config, save=False, output_dir=None):# n_iter, lr, n_batch=1024, beta_schedule=None, n_checkpoints=10):

    if save:
        with open(output_dir+'/config.json', 'x') as config_file:
            json.dump(config, config_file, indent=4)
    n_iter = config['opt_param']['n_iter']
    n_batch = config['opt_param']['n_batch']
    n_checkpoints = config['opt_param']['n_checkpoints']
    lr = config['opt_param']['learning_rate']

    beta_schedule = annealing_schedule(np.arange(n_iter), config['annealing_param'])
    radius = torch.norm(target.means[0]).item()

    #metrics
    loss_list=[]
    vars = [[],[]]
    summary_statistics=[[],[],[]]
    betas = []

    #checkpoints
    if save:
        torch.save(model.state_dict(), output_dir+'/checkpoint0.pt')

    torch.manual_seed(config['opt_param']['seed'])

    for t in range(n_iter):
        beta = beta_schedule[t]

        z_sample = model.sample(n_batch)
        loss = torch.mean(beta*target.nll(z_sample) - model.nll(z_sample))
        loss.backward()

        #jko scheme
        with torch.no_grad():
            model.means -= lr/beta * model.means.grad
            model.vars *= ( 1 - 2*lr/beta/model.dim*model.vars.grad )**2
            model.means.data *= radius / (model.means.data**2).sum(dim=1)[:, None]**(1/2)
        
        #metrics
        with torch.no_grad():
            log_Z = torch.logsumexp(np.log(2*radius/100) + beta*(target.log_marginal_density(torch.linspace(-10*radius, 10*radius, 1000, device=model.device))), axis=-1) - (target.dim-1)/2*(np.log(beta) + (beta-1)*np.log(2*np.pi))
            loss_list.append((loss + log_Z).item())

            #mean_grad_norms.append(model.means.grad.detach().norm(2))
            #var_grad_norms.append(model.vars.grad.detach().norm(2))
            vars[0].append(model.vars[0].item())
            vars[1].append(model.vars[1].item())
            summary_statistics[0].append(model.means[0,0].item()/radius)
            summary_statistics[1].append(model.means[1,0].item()/radius)
            summary_statistics[2].append(torch.sum(model.means[0].detach()*model.means[1].detach()).item() / radius**2)
            betas.append(beta)

        model.means.grad.zero_()
        model.vars.grad.zero_()
        
        #checkpoints
        if save:
            if (t+1) % (n_iter//n_checkpoints) == 0:
                torch.save(model.state_dict(), output_dir+'/checkpoint'+str((t+1) // (n_iter//n_checkpoints))+'.pt')
        

    
    to_return = {
        'losses' : loss_list,
        'var1': vars[0],
        'var2': vars[1],
        'm1': summary_statistics[0],
        'm2': summary_statistics[1],
        's': summary_statistics[2],
        'betas' : betas,
    }

    if save:
        with open(output_dir+'/metrics.json', 'x') as metrics_file:
            json.dump(to_return, metrics_file, indent=4)

    return to_return