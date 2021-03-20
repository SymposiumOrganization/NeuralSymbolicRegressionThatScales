import os
import numpy as np
import random
import math
from scipy.optimize import minimize
import types
import click
import marshal
import copyreg
import sys
import ast
import pdb
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
import torch.nn.functional as F
import sympy as sp
from dataclasses import dataclass
from ..dataset.generator import Generator
from . import data
from typing import Tuple
import time
import re
from ..dataset.sympy_utils import add_multiplicative_constants, add_additive_constants



class TimedFun:
    def __init__(self, fun, stop_after=10):
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after

    def fun(self, x, *args):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        self.fun_value = self.fun_in(*x, *args)
        self.x = x
        return self.fun_value


def bfgs(pred_str, X, cfg):#n_restarts, env, NMSE=True, idx_remove =True, normalization_o= False):
    #Check where dimensions not use, and replace them with 1.
    X = X.clone()
    bool_dim = (X==0).all(axis=2).squeeze()
    X[:,bool_dim,:] = 1
    # first_mask_step = [torch.sum(X[0,:,i] == 0) for i in range(X.shape[2])]
    # mask = [(first_mask_step[j] == X.shape[1]).numpy() for j in range(X.shape[2])]
    # x_bfgs = X.clone()
    # for i in range(X.shape[2]):
    #     if mask[i] == True:
    #         x_bfgs[:,:,i] = X[:,:,i]+1 #BFGS wants 1 for a non existing variable 
    # bfgs_input = torch.cat((x_bfgs, y), dim=1)
    pred_str = pred_str[1:].tolist()
    #pred_str = [x if x<14 else x+1 for x in pred_str]
    prefix = data.de_tokenize(pred_str, cfg.id2word)
    # if "constant" in prefix:
    #     for j,i in enumerate(list(pred_str)[:-1]):
    #         if i == "constant":    
    #             expre[j] = 'c{}'.format(str(c))
    #             c=c+1
    #     example = "".join(list(expre))  

    if cfg.bfgs.add_coefficients_if_not_existing and 'constant' not in prefix:           
        print("No constants in predicted expression. Attaching them everywhere")
        variables = {x:sp.Symbol(x, real=True, nonzero=True) for x in cfg.total_variables}
        infix = Generator.prefix_to_infix(prefix, coefficients=cfg.total_coefficients, variables=cfg.total_variables)
        s = Generator.infix_to_sympy(infix,variables, cfg.rewrite_functions)
        placeholder = {x:sp.Symbol(x, real=True,nonzero=True) for x in ["cm","ca"]}
        s = add_multiplicative_constants(s, placeholder["cm"], unary_operators=cfg.una_ops)
        s = add_additive_constants(s,  placeholder, unary_operators=cfg.una_ops)
        s = s.subs(placeholder["cm"],0.43)
        s = s.subs(placeholder["ca"],0.421)
        s_simplified = data.constants_to_placeholder(s,symbol="constant")
        prefix = Generator.sympy_to_prefix(s_simplified)
        # prefix = ["constant" if c in ["cm","ca"] else c for c in prefix ]
        # temp = env.sympy_to_prefix(sympify(pred_str))
        # temp2 = env._prefix_to_infix_with_constants(temp)[0]
        # num = env.count_number_of_constants(temp2)
        # costs = [random.random() for x in range(num)]
        # example = temp2.format(*tuple(costs))
        # pred_str = str(env.constants_to_placeholder(example))
        # c=0
        # expre = list(pred_str)
    
    # candidate = Generator.prefix_to_infix(prefix, 
    #                                 coefficients=["constant"], 
    #                                 variables=cfg.total_variables)
    
    
    breakpoint()
    for j,i in enumerate(list(candidate)[:-1]):
        if i == "constant":    
            candidate[j] = 'c{}'.format(str(c))
            c=c+1
    example = "".join(list(candidate))  
    # for j,i in enumerate(list(pred_str)):
    #     try:
    #         if i == 'c' and list(pred_str)[j+1] != 'o':
    #             expre[j] = 'c{}'.format(str(c))
    #             c=c+1
    #     except IndexError:
    #         if i == 'c':
    #             expre[j] = 'c{}'.format(str(c))
    #             c=c+1        
    # example = "".join(list(expre))


    
    
    print('Constructing BFGS loss...')
    #construct loss function
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    symbols = {}
    for i in range(0,40):                                        #change to actual number of consts
        symbols[i] = Symbol('c{}'.format(i))
    input_batch = input_batch.cpu() 
    xx = input_batch[:,: len(cfg.total_variables)].T  

    cfg.
    if idx_remove:
        print('Removing indeces with high values...')
        idx_leave = np.where((np.abs(input_batch[:,3].numpy()))<200)[0]
        xx = xx[:,idx_leave]
        input_batch = input_batch[idx_leave,:]

    mean = (np.mean(torch.abs(input_batch[:,-1]).numpy()))
    max = np.max(np.abs(torch.abs(input_batch[:,-1]).numpy()))
    print('checking input values range...')
    if mean >100 or max > 300:
        print('Attention, input values are very large. Optimization may fail due to numerical issues')
            
    diff = [sympify(example).replace(y,xx[1,i]).replace(x,xx[0,i]).replace(z,xx[2,i])-input_batch[i,-1] for i in range(input_batch.shape[0])]
    if normalization_o:
        diff = [sympify(example).replace(y,xx[1,i]).replace(x,xx[0,i]).replace(z,xx[2,i])-input_batch[i,-1]/max for i in range(input_batch.shape[0])]
    loss = 0
    cnt = 0
    if NMSE == True and (mean != 0):
        loss = (np.mean(np.square(diff)))/mean  ###3 check
    else: 
        loss = (np.mean(np.square(diff)))
    
    print('Loss constructed, starting BFGS optmization...') 
    #bfgs optimization
    F_loss = []
    consts_ = []
    funcs = []
    for i in range(n_restarts):
        print('BFGS optimization: iteration # ', i)
        n_symbols = len(loss.atoms(Symbol))
        x0 = np.random.randn(n_symbols)
        s = []
        for i in range(0,n_symbols):
            s.append((symbols[i]))
        fun_timed = TimedFun(fun=lambdify(s,loss, modules=['numpy']), stop_after=1e9)
        minimize(fun_timed.fun,x0, method='BFGS')   #check consts interval and if they are int
        consts_.append(fun_timed.x)
        final = example
        for i in range(len(s)):
            final = sympify(final).replace(s[i],fun_timed.x[i])
        if normalization_o:
            funcs.append(max*final)
        else:
            funcs.append(final)
        final_loss = np.mean(np.square(lambdify('x,y,z', final)(*xx)-input_batch[:,-1]).numpy())
        F_loss.append(final_loss)
         #early stopping
        if final_loss < 1e-8:
            return (final, fun_timed.x,final_loss,example)
    k_best = np.nanargmin(F_loss)
    return funcs[k_best], consts_[k_best], F_loss[k_best], example
            