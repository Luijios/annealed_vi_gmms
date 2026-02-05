import argparse
import torch
import numpy as np


from isotropic_gaussian import *
from train_bimodal import *
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

target_means = torch.zeros(2, dim, device=device)
target_means[0,0] = radius
target_means[1,0] = -radius
target_vars = torch.tensor([1., 1.], device=device)
target_weights = torch.tensor([w_star, 1-w_star], device=device)
target = MIG(target_means, target_vars, target_weights, device=device)


##### MODEL ######
beta_i = config['annealing_param']['beta_i']
w_1 = config['model_param']['weight']
torch.manual_seed(config['model_param']['seed'])

means = nn.Parameter(torch.randn(2, dim, device=device))
means.data = radius * (means.data / ((means.data**2).sum(dim=1)[:,None])**0.5)
vars = nn.Parameter(torch.tensor([1/beta_i, 1/beta_i], device=device))
weights = torch.tensor([w_1, 1-w_1], device=device)
model = MIG(means, vars, weights, device=device)


##### THE TRAINING ######
_ = train_bimodal(model, target, config, save=True, output_dir=args.output_dir)