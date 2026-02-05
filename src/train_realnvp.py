import torch
import numpy as np
from utils import *
import json


############################
###### TRAINING LOOP #######
############################
    
def train_realnvp(flow, target, config, save=False, output_dir=None):
    if save:
        with open(output_dir+'/config.json', 'x') as config_file:
            json.dump(config, config_file, indent=4)
    n_iter = config['opt_param']['n_iter']
    n_batch = config['opt_param']['n_batch']
    n_checkpoints = config['opt_param']['n_checkpoints']

    beta_schedule = annealing_schedule(np.arange(n_iter), config['annealing_param'])
    radius = torch.norm(target.means[0]).item()
    x_range = torch.linspace(-10*radius, 10*radius, 2000, device=target.device)
    log_marginal = log_marginal_along_mu(radius, torch.tensor([1,-1], device=target.device), target.vars, target.weights, x_range)

    opt = torch.optim.Adam([
        {'params': flow.affine_couplings.parameters()},
        #{'params': flow.batchnorm_layers[:-1].parameters()},
        #{'params': flow.batchnorm_layers[-1].parameters(), 'lr': 1e-1},
    ], lr=config['opt_param']['learning_rate'])

    #metrics
    loss_list = []
    means = []
    vars = []
    param_norms = [[] for k in range(flow.num_layers)]
    param_grad_norms = [[] for k in range(flow.num_layers)]

    #checkpoints
    if save:
        torch.save(flow.state_dict(), output_dir+'/checkpoint0.pt')

    torch.manual_seed(config['opt_param']['seed'])
    flow.train()

    for t in range(n_iter):
        beta = beta_schedule[t]
        opt.zero_grad()

        z = flow.prior.sample((n_batch,))
        x, log_det = flow(z)
        loss = torch.mean(-log_det + beta*target.nll(x))

        loss.backward()
        opt.step()

        #logs
        with torch.no_grad():
            #log_Z = torch.logsumexp(np.log(2*radius/100) + beta*(target.log_marginal_density(torch.linspace(-10*radius, 10*radius, 1000, device=flow.device))), axis=-1) - (target.dim-1)/2*(np.log(beta) + (beta-1)*np.log(2*np.pi))
            #means.append(torch.mean(x[:,0].detach(), axis=0).item())
            #vars.append(torch.std(x[:,0].detach(), axis=0).item())

            log_Z = torch.logsumexp(np.log(radius/100) + beta*log_marginal, axis=-1) - (target.dim-1)/2*(np.log(beta) + (beta-1)*np.log(2*np.pi))
            loss_list.append((loss + log_Z + torch.mean(flow.prior.log_prob(z))).item())

            proj = torch.sum(target.means[0]*x.detach(), dim=1)
            means.append(torch.mean(proj).item()/radius)
            vars.append(torch.std(proj).item()/radius)
        
            for k in range(flow.num_layers):
                norm=[]
                grad_norm=[]
                for p in flow.affine_couplings[k].parameters():
                    if p.requires_grad:
                        norm.append(torch.norm(p.detach()).item())
                        grad_norm.append(torch.norm(p.grad.detach()).item())
                param_grad_norms[k].append(np.mean(grad_norm))
                param_norms[k].append(np.mean(norm))
            
        #checkpoints
        if save:
            if (t+1) % (n_iter//n_checkpoints) == 0:
                torch.save(flow.state_dict(), output_dir+'/checkpoint'+str((t+1) // (n_iter//n_checkpoints))+'.pt')
                #models.append(copy.deepcopy(flow).to('cpu'))

    flow.eval()
    
    to_return = {
        'losses' : loss_list,
        'means' : means,
        'vars' : vars,
        'param_norms' : param_norms,
        'param_grad_norms' : param_grad_norms,
        'betas' : beta_schedule.tolist(),
    }

    if save:
        with open(output_dir+'/metrics.json', 'x') as metrics_file:
            json.dump(to_return, metrics_file, indent=4)

    return to_return
