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

def pre_forward_sampling_hook(m, i):
    for key in m._variational_parameters:
        if DEBUG_INFO: print('Sampling ' + key)
        setattr(m, key, m._distributions['posterior_%s' % key].rsample())

    for child in m.children():
        for key in child._variational_parameters:
            if DEBUG_INFO: print('Sampling ' + key)
            setattr(child, key, child._distributions['posterior_%s' % key].rsample())

def forward_sampling_hook(m, i, o):
    if not m._sampling:
        m._sampling = True
        n = m._n_samples
        o_samples = torch.stack([o] + [m(*i) for _ in range(n - 1)])
        o = o_samples.mean(0)
        m._uncertainity = o_samples.std(0)
        m._sampling = False
        
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

def bayesify(l, n_samples=10):
    l._n_samples = n_samples
    l.register_forward_pre_hook(pre_forward_sampling_hook)
    l.register_forward_hook(forward_sampling_hook)
    _bayesify(l)

def _bayesify(l):
    keys   = []
    shapes = []
    l._sampling = False
    if not hasattr(l, "_variational_parameters"):
        l._variational_parameters = []
        l._distributions = {}
        l.get_var_cost = MethodType(get_var_cost, l)

    for key in list(l._parameters.keys()):
        if not is_variational_parameter(l, key):
            bayesify_parameter(l, key)

    for child in l.children():
        _bayesify(child)
