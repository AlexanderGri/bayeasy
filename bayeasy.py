import torch
from torch import nn
from torch.distributions.normal import Normal
from types import MethodType
import numpy as np


DEBUG_INFO = False

param_prefixes = ['_prior_loc_', '_prior_scale_', '_posterior_loc_', '_posterior_scale_']

def is_variational_parameter(l, key):

    if key in l._variational_parameters:
        return True

    for prefix in param_prefixes:
        if key.startswith(prefix) and key[len(prefix):] in l._variational_parameters:
            return True
        
    return False

def bayesify_parameter(l, key):
    if is_variational_parameter(l, key):
        return
    
    value = getattr(l, key)
    shape = value.data.shape
    
    loc, scale = (0., 1.)

    val = l._parameters[key].data
    del l._parameters[key]

    l.register_buffer(key, val)
    l.register_buffer('_prior_loc_%s' % key, loc * torch.ones(*shape))
    l.register_buffer('_prior_scale_%s' % key, scale * torch.ones(*shape))
    l._distributions['prior_%s' % key] = Normal(l._buffers['_prior_loc_%s' % key],
                                                l._buffers['_prior_scale_%s' % key])

    
    l.register_parameter('_posterior_loc_%s' % key, nn.Parameter((loc * torch.ones(*shape)).requires_grad_()))
    l.register_parameter('_posterior_scale_%s' % key, nn.Parameter((scale * torch.ones(*shape)).requires_grad_()))
    l._distributions['posterior_%s' % key] = Normal(l._parameters['_posterior_loc_%s' % key],
                                                    l._parameters['_posterior_scale_%s' % key])

    l._variational_parameters.append(key)

    if DEBUG_INFO: print('Parameter %s in %s is bayesified' % (key, str(l)))

def sampling_hook(m, i):
    for key in m._variational_parameters:
        if DEBUG_INFO: print('Sampling ' + key)
        setattr(m, key, m._distributions['posterior_%s' % key].rsample())
        
def get_var_cost(m):
    # Stochastic Gradient Langevin Dynamics

    var_cost = torch.tensor(0.)

    if hasattr(m, '_variational_parameters'):    
        for key in m._variational_parameters:
            sample = getattr(m, key)
            prior_distr = m._distributions['prior_%s' % key]
            var_cost = var_cost + prior_distr.log_prob(sample).sum() # regularizer

    for child in m.children():
        if hasattr(child, 'get_var_cost'):
            var_cost = var_cost + get_var_cost(child)
    
    return var_cost

def bayesify(l):
    keys   = []
    shapes = []
    if not hasattr(l, "_variational_parameters"):
        l._variational_parameters = []
        l._distributions = {}
        l.register_forward_pre_hook(sampling_hook)
        l.get_var_cost = MethodType(get_var_cost, l)

    for key in list(l._parameters.keys()):
        if not is_variational_parameter(l, key):
            bayesify_parameter(l, key)

    for child in l.children():
        bayesify(child)
