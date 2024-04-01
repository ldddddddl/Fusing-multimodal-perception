# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 15:00:57 2023

@author: admin
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal


def duplicate(x, rep, dim=0):
    if dim == 0:
        return x.expand(rep, *x.shape).reshape(-1, *x.shape[1:])
    elif dim == -1:
        return x.expand(rep, *x.shape).reshape(*x.shape[:-1], -1)
        

def sample_gaussian(m, v, device, training_phase):
    if training_phase == 'generate':
        m = torch.zeros_like(m)  #zero
        v = torch.ones_like(v)   #one
    epsilon = Normal(0, 1).sample(m.size())
    z = m + torch.sqrt(v) * epsilon.to(device)  

    return z

def gaussian_parameters(h, dim=-1):

    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def product_of_experts(m_vect, v_vect, flag):


    '''
    增加了一个很小的数避免出现分母为零
    '''
    if flag == 'grasping':  
        mu = m_vect
        var = v_vect
    elif flag == 'generate':
        T_vect = pow(v_vect, -2) + 1e-08
        mu = (m_vect * T_vect).sum(-1) * (1.0 / T_vect.sum(-1)) + 1e-08 #torch.Size([8, 128])
        var = 1.0 / T_vect.sum(-1) + 1e-08  #torch.Size([8, 128])
    
    return mu, var