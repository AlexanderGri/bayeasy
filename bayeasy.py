import torch
from torch import nn
from torch.distributions.normal import Normal
from types import MethodType
import numpy as np


DEBUG_INFO = False

param_prefixes = ['_prior_loc_', '_prior_scale_', '_posterior_loc_', '_posterior_scale_']

def is_variational_parameter(model, key):
    """Check whether parameter key is a variational parameter of model. 

    Args:
        key: name of the parameter
    
    Returns:
        bool: True if variational. False otherwise.

    """
    if key in model._variational_parameters:
        return True

    for prefix in param_prefixes:
        if key.startswith(prefix) and key[len(prefix):] in model._variational_parameters:
            return True
        
    return False

def bayesify_parameter(model, key):
    """Turn parameter key of model to random variable

    Args:
        key: name of the parameter
        model: pytorch module

    Returns:
        None

    """
    if is_variational_parameter(model, key):
        return
    
    value = getattr(model, key)
    shape = value.data.shape
    
    loc, scale = (0., 1.)

    val = model._parameters[key].data
    del model._parameters[key]

    model.register_buffer(key, val)
    model.register_buffer('_prior_loc_%s' % key, loc * torch.ones(*shape))
    model.register_buffer('_prior_scale_%s' % key, scale * torch.ones(*shape))
    model._distributions['prior_%s' % key] = Normal(model._buffers['_prior_loc_%s' % key],
                                                    model._buffers['_prior_scale_%s' % key])

    
    model.register_parameter('_posterior_loc_%s' % key, 
                             nn.Parameter((loc * torch.ones(*shape)).requires_grad_()))
    model.register_parameter('_posterior_scale_%s' % key,
                             nn.Parameter((scale * torch.ones(*shape)).requires_grad_()))
    model._distributions['posterior_%s' % key] = Normal(model._parameters['_posterior_loc_%s' % key],
                                                        model._parameters['_posterior_scale_%s' % key])

    model._variational_parameters.append(key)

    if DEBUG_INFO: print('Parameter %s in %s is bayesified' % (key, str(model)))

def pre_forward_sampling_hook(model, input):
    '''Pytorch hook for sampling model parameters before computation

    Args:
        model: pytorch module
        input: input tensor

    Returns:
        None

    '''
    for key in model._variational_parameters:
        if DEBUG_INFO: print('Sampling ' + key)
        setattr(model, key, model._distributions['posterior_%s' % key].rsample())

    for child in model.children():
        for key in child._variational_parameters:
            if DEBUG_INFO: print('Sampling ' + key)
            setattr(child, key, child._distributions['posterior_%s' % key].rsample())

def forward_sampling_hook(model, input, output):
    '''Pytorch hook for sampling model parameters before computation

    Args:
        model: pytorch module
        input: input tensor 
        output: output tensor after forward

    Returns:
        None

    '''
    if not model._sampling:
        model._sampling = True
        n = model._n_samples
        output_samples = torch.stack([output] + [model(*input) for _ in range(n - 1)])
        output = output_samples.mean(0)
        model._uncertainity = output_samples.std(0)
        model._sampling = False
        
def get_var_cost(model):
    '''Regularizer of variational cost according to Stochastic Gradient Langevine Dynamics

    Args:
        model: pytorch module

    Returns:
        torch.tensor: variational addition to loss

    '''

    var_cost = torch.tensor(0.)

    if hasattr(model, '_variational_parameters'):    
        for key in model._variational_parameters:
            sample = getattr(model, key)
            prior_distr = model._distributions['prior_%s' % key]
            var_cost = var_cost + prior_distr.log_prob(sample).sum() # regularizer

    for child in model.children():
        if hasattr(child, 'get_var_cost'):
            var_cost = var_cost + get_var_cost(child)
    
    return var_cost

def bayesify(model, n_samples=10):
    '''Transform module parameters to random gaussian variables which are sampled during forward.
    
    Args:
        model: pytorch module
        n_samples: int, adjusts number of samples during Monte Carlo estimation of output

    Returns:
        None

    '''
    model._n_samples = n_samples
    model.register_forward_pre_hook(pre_forward_sampling_hook)
    model.register_forward_hook(forward_sampling_hook)
    _bayesify(model)

def _bayesify(model):
    '''Helper function'''
    keys   = []
    shapes = []
    model._sampling = False
    if not hasattr(model, "_variational_parameters"):
        model._variational_parameters = []
        model._distributions = {}
        model.get_var_cost = MethodType(get_var_cost, model)

    for key in list(model._parameters.keys()):
        if not is_variational_parameter(model, key):
            bayesify_parameter(model, key)

    for child in model.children():
        _bayesify(child)
