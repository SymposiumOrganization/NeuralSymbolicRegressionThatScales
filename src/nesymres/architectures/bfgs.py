import os
import numpy as np
import random
import math
from torch.utils.data import TensorDataset, DataLoader,Dataset
from eq_learner import utils
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
from sympy import *
from dataclasses import dataclass
from typing import Tuple
import time
import re



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


def bfgs(pred_str, input_batch, n_restarts, env, NMSE=True, idx_remove =True, normalization_o= False):
    #use different constant symbols to treat each variable separately
    c=0
    expre = list(pred_str)
    for j,i in enumerate(list(pred_str)[:-1]):
        if i == 'c' and list(pred_str)[j-1] == '(' and list(pred_str)[j+1] == ')':    ##### Check later
            expre[j] = 'c{}'.format(str(c))
            c=c+1
    example = "".join(list(expre))  
    
    
    if 'c0' not in example:                                                           ##### make a flag
        print("No constants in predicted expression. Attaching them everywhere")
        temp = env.sympy_to_prefix(sympify(pred_str))
        temp2 = env._prefix_to_infix_with_constants(temp)[0]
        num = env.count_number_of_constants(temp2)
        costs = [random.random() for x in range(num)]
        example = temp2.format(*tuple(costs))
        pred_str = str(env.constants_to_placeholder(example))
        c=0
        expre = list(pred_str)
        for j,i in enumerate(list(pred_str)):
            try:
                if i == 'c' and list(pred_str)[j+1] != 'o':
                    expre[j] = 'c{}'.format(str(c))
                    c=c+1
            except IndexError:
                if i == 'c':
                    expre[j] = 'c{}'.format(str(c))
                    c=c+1        
        example = "".join(list(expre))
    
    print('Constructing BFGS loss...')
    #construct loss function
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    symbols = {}
    for i in range(0,40):                                        #change to actual number of consts
        symbols[i] = Symbol('c{}'.format(i))
    input_batch = input_batch.cpu() 
    xx = input_batch[:,:3].T  

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
            