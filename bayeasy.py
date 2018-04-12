import torch
from torch.autograd import Variable as Var
from torch import nn
from types import MethodType
import numpy as np


prefixes = ['mu_', 'logsigma_']
log2pi = float(np.log(2 * np.pi))

def log_normal(x, mean, log_std):
    return - 0.5 * log2pi - log_std - (x - mean) ** 2 / (2 * torch.exp(2 * log_std))

def is_valid_parameter(l, key):

    if key in l.variational_parameters:
        return False

    for prefix in prefixes:
        if key.startswith(prefix) and key[len(prefix):] in l.variational_parameters:
            return False
        
    return True

def bayesify_parameter(l, key):
    if not hasattr(l, 'variational_parameters'):
        l.variational_parameters = []
        
    if not is_valid_parameter(l, key):
        return
    
    value = getattr(l, key)
    shape = value.data.shape
    l.register_buffer('sample_for_' + key, nn.Parameter(torch.randn(*shape)))

    l.register_buffer('prior_mu_' + key, nn.Parameter(torch.zeros(*shape)))
    l.register_buffer('prior_logsigma_' + key, nn.Parameter(torch.zeros(*shape)))

    l.register_parameter('mu_' + key, nn.Parameter(torch.randn(*shape)))
    l.register_parameter('logsigma_' + key, nn.Parameter(torch.randn(*shape)))
    
    l.variational_parameters.append(key)
    
    print('Parameter %s in %s is bayesified' % (key, str(l)))

def sampling_hook(m, i):
    for key in m.variational_parameters:
        print('Sampling ' + key)
        name = 'sample_for_' + key
        sample = Var(torch.randn(*m._buffers[name].data.shape))
        m._buffers[name] = sample
        mu = getattr(m, 'mu_' + key)
        logsigma = getattr(m, 'logsigma_' + key)
        m._parameters[key] = mu + torch.exp(logsigma) * sample
        
def get_var_cost(model):
    var_cost = Var(torch.zeros(1), requires_grad=True)
    
    if hasattr(model, 'variational_parameters'):    
        for key in model.variational_parameters:
            sample = getattr(model, key)

            prior_mu = getattr(model, 'prior_mu_' + key)
            mu = getattr(model, 'mu_' + key)

            prior_logsigma = getattr(model, 'prior_logsigma_' + key)
            logsigma = getattr(model, 'logsigma_' + key)

            var_cost = var_cost + torch.sum(log_normal(sample, mu, logsigma) - log_normal(sample, prior_mu, prior_logsigma))

    for child in model.children():
        if hasattr(child, 'get_var_cost'):
            var_cost = var_cost + get_var_cost(child)
    
    return var_cost

def bayesify(l):
    keys   = []
    shapes = []
    for key in list(l._parameters.keys()):
        bayesify_parameter(l, key)
    l.register_forward_pre_hook(sampling_hook)
    l.get_var_cost = MethodType(get_var_cost, l)
    
    for child in l.children():
        bayesify(child)
