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


def evaluate_and_wrap(
    functions,
    symbols,
    tokens,
    p,
    type_of_sampling_points,
    support_distribution,
    const_dist,
):
    vals = []
    cond0 = []
    curr_p = return_support(p, type_of_sampling_points)
    mode = inspect.getargspec(functions[0])
    for i in range(len(functions)):
        if int(tokens[i][0]) == 0:
            #           breakpoint()
            cond0.append(False)
            continue
        sym = {}
        for idx, sy in enumerate(["x", "y", "z"]):
            if sy in symbols[i]:
                sym[idx] = support_distribution[i][idx].sample([int(curr_p)])
            else:
                sym[idx] = torch.zeros(int(curr_p))
        consts = torch.stack(
            [
                torch.ones([int(curr_p)]) * const_dist[i][c].sample()
                for c in ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]
            ]
        )
        support = torch.stack([sym[0], sym[1], sym[2]])
        # breakpoint()
        if len(mode.args) > 3:
            input_lambdi = torch.cat([support, consts], axis=0)
        else:
            input_lambdi = support
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = functions[i]
                aaaa = f(*input_lambdi)
                if type(aaaa) == torch.Tensor and aaaa.dtype == torch.float32:
                    vals.append(
                        torch.cat(
                            [support, torch.unsqueeze(aaaa, axis=0)], axis=0
                        ).unsqueeze(0)
                    )
                    cond0.append(True)
                else:
                    cond0.append(False)
        except NameError as e:
            # print(e)
            cond0.append(False)
        # except:
        #     breakpoint()
    tokens = tokens[cond0]
    num_tensors = torch.cat(vals, axis=0)
    cond = (
        torch.sum(torch.count_nonzero(torch.isnan(num_tensors), dim=2), dim=1)
        < curr_p / 25
    )
    num_fil_nan = num_tensors[cond]
    tokens = tokens[cond]
    cond2 = (
        torch.sum(
            torch.count_nonzero(torch.abs(num_fil_nan) > 5e4, dim=2), dim=1
        )  # Luca comment 0n 21/01
        < curr_p / 25
    )
    num_fil_nan_big = num_fil_nan[cond2]
    tokens = tokens[cond2]
    idx = torch.argsort(num_fil_nan_big[:, 3, :]).unsqueeze(1).repeat(1, 4, 1)
    res = torch.gather(num_fil_nan_big, 2, idx)
    # res, _ = torch.sort(num_fil_nan_big)
    res = res[:, :, torch.sum(torch.count_nonzero(torch.isnan(res), dim=1), dim=0) == 0]
    res = res[
        :,
        :,
        torch.sum(torch.count_nonzero(torch.abs(res) > 5e4, dim=1), dim=0)
        == 0,  # Luca comment 0n 21/01
    ]
    return res, tokens
    # num_fil_nan = num_tensors[torch.sum(torch.count_nonzero(torch.isnan(num_tensors), dim=2),dim=1) < p/10]


def tokens_padding(tokens):
    # tokens = [x for idx, x in enumerate(list(zip(*y))[1])]
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        p_tokens[i, :] = torch.cat([y.long(), torch.zeros(max_len - y.shape[0]).long()])
    return p_tokens
    # breakpoint()

    # #import pdb; pdb.set_trace()
    # if type(aaaa) == int or type(aaaa) == float or type(aaaa) == complex or type(aaaa) == np.float64 or not torch.float32 == aaaa.dtype:
    #     return None
    # elif torch.isnan(aaaa).any():
    #     return None
    # elif (torch.abs(aaaa)>1e3).any():
    #     return None
    # res = torch.cat([support,torch.unsqueeze(aaaa,axis=0)],axis=0)
    # return res



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