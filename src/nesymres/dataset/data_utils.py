import torch
import torch.nn as nn
from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
import warnings
import inspect
from torch.distributions.uniform import Uniform
import math
import types
from numpy import log, cosh, sinh, exp, cos, tanh, sqrt, sin, tan, arctan, nan, pi, e, arcsin, arccos
from sympy import sympify, Symbol
from sympy import Float


def dataset_loader(train_dataset, test_dataset, batch_size=1024, valid_size=0.20):
    num_train = len(train_dataset)
    num_test_h = len(test_dataset)
    indices = list(range(num_train))
    test_idx_h = list(range(num_test_h))
    np.random.shuffle(test_idx_h)
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=0
    )
    test_loader_h = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, valid_loader, test_loader_h, valid_idx, train_idx


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def initialize_weights(m):
    """Used for the transformer"""
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def return_support(p, type_of_sampling_points):
    if type_of_sampling_points[0] == "Constant":
        curr_p = min(p)
    elif type_of_sampling_points[0] == "Logarithm":
        curr_p = int(10 ** Uniform(1, math.log10(min(p))).sample())
    else:
        raise NameError
    return curr_p

def evaluate_fun(args):
    fun ,support = args
    if type(fun)==list and not len(fun):
        return []
    f = types.FunctionType(fun, globals=globals(), name='f')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            evaled = f(*support)
            if type(evaled) == torch.Tensor and evaled.dtype == torch.float32:
                return evaled.numpy().astype('float16')
            else:
                return []
    except NameError as e:
        print(e)
        return []

