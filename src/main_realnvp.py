import argparse
import json
import torch
import numpy as np

from isotropic_gaussian import *
from train_realnvp import *
from realnvpwithscale import *

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')

parser = argparse.ArgumentParser(description='Prepare experiment')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--output-dir', type=str, required=True)
args = parser.parse_args()

with open(args.config, 'r') as file:
    config = json.load(file)


##### TARGET ######
dim = config['teacher_param']['dim']
radius = config['teacher_param']['radius']
w_star = config['teacher_param']['weight']

torch.manual_seed(config['teacher_param']['seed'])
mu_star = torch.randn(dim, device=device)
mu_star = mu_star / torch.norm(mu_star)
target_means = torch.stack([radius*mu_star, -radius*mu_star])
target_vars = torch.tensor([1., 1.], device=device)
target_weights = torch.tensor([w_star, 1-w_star], device=device)
target = MIG(target_means, target_vars, target_weights, device=device)


##### MODEL ######
torch.manual_seed(config['nf_param']['seed'])
flow = RealNVP(config['nf_param']['num_layers'], dim, config['nf_param']['hidden_dim'], device=device)


##### THE TRAINING ######
_ = train_realnvp(flow, target, config, save=True, output_dir=args.output_dir)

