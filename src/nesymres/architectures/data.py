import numpy as np
import sympy
import warnings
import torch
from torch.utils import data
import math
import pandas as pd
from dataclasses import dataclass
from sympy.core.rules import Transform
from sympy import sympify
from sympy import trigsimp
from sympy import Float, Symbol
import pdb
from multiprocessing import Manager
from numpy import (
    log,
    cosh,
    sinh,
    exp,
    cos,
    tanh,
    sqrt,
    sin,
    tan,
    arctan,
    nan,
    pi,
    e,
    arcsin,
    arccos,
)

import types
from typing import List
import random
from torch.distributions.uniform import Uniform
import random
from ..dclasses import DataModuleParams
from ..dataset.generator import Generator
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from ..utils import load_data
from nesymres.dclasses import Params, Dataset, Equation

class NesymresDataset(data.Dataset):
    def __init__(
        self,
        data: Dataset,
        data_params: DataModuleParams,
    ):  
        m = Manager()
        self.eqs = m.dict({i:eq for i, eq in enumerate(data.eqs)})
        self.data_params = data_params
        self.data = data

    def __getitem__(self, index):
        eq = self.eqs[index]
        code = types.FunctionType(eq.code, globals=globals(), name="f")
        consts = {const: 1 if const[:2] == "cm" else 0 for const in eq.coeff_dict.keys()}
        if self.data_params.predict_c:
            used_consts = random.randint(0, min(len(eq.coeff_dict),self.data_params.constant_degree_of_freedom))
            symbols_used = random.sample(set(eq.coeff_dict.keys()), used_consts)
            for si in symbols_used:
                consts[si] = Uniform(self.data_params.constant_support[0], self.data_params.constant_support[1]).sample()

        eq_string = eq.expr.format(**consts)
        eq_sympy_infix = constants_to_placeholder(eq_string)
        eq_sympy_prefix = Generator.sympy_to_prefix(eq_sympy_infix)

        if not self.data_params.predict_c:
            assert "c" not in eq_sympy_prefix
        try:
            t = tokenize(eq_sympy_prefix,self.data.word2id)
            curr = Equation(code=code,expr=eq_string,coeff_dict=eq.coeff_dict,variables=eq.variables,support=eq.support, tokenized=t, valid=True)
        except:
            t = []
            curr = Equation(code=code,expr=eq_string,coeff_dict=eq.coeff_dict,variables=eq.variables,support=eq.support, valid=False)
        return curr
        # costs = [
        #     random.random()
        #     if not bool(total_consts_symbols[x].low == total_consts_symbols[x].high)
        #     else int(total_consts_symbols[x].low)
        #     for x in available_constans_symbols
        # ]
        # if self.data_params.predict_c:
        #     example = eq.expr.format(**consts)
            
        #     # if self.input_normalization:
        #     #     sympy_expr = self.env.constants_to_placeholder(example)*Symbol('c') + Symbol('c')
        #     #sympy_expr = trigsimp(sympy_expr)  # Delete if slow
        #     try:
        #         prefix = self.env.sympy_to_prefix(sympy_expr)
        #         t = self.env.tokenize(prefix)
        #         tokens = torch.tensor(t)
        #     except:
        #         #print("Error with {}".format(sympy_expr))  # lUCA COMMENT
        #         tokens = torch.tensor([0, 0, 0, 0, 0])
        #         # breakpoint()
        # support = []
        # min_supp_len = 2
        # for i in range(len(symbols)):
        #     mi = np.random.uniform(self.support_extremes[0], self.support_extremes[1]-min_supp_len)
        #     ma = np.random.uniform(mi+min_supp_len, self.support_extremes[1])
        #     support.append(Uniform(mi, ma))
        
        #     f,
        #     symbols,
        #     tokens,
        #     self.p[index],
        #     self.type_of_sampling_points,
        #     support,
        #     total_consts_symbols,
        # )

    def __len__(self):
        return len(self.eqs)



def custom_collate_fn(eqs: List[Equation]):
    filtered_eqs = [eq for eq in eqs if eq.valid ]
    tokens_eqs = [eq.tokenized for eq in eqs]
    tokens_eqs = tokens_padding(tokens_eqs)

    res, tokens = evaluate_and_wrap(filtered_eqs)


    return res, tokens


def constants_to_placeholder(s):
    try:
        sympy_expr = sympify(s)  # self.infix_to_sympy("(" + s + ")")
        sympy_expr = sympy_expr.xreplace(
            Transform(
                lambda x: Symbol("c", real=True, nonzero=True),
                lambda x: isinstance(x, Float),
            )
        )
    except:
        breakpoint()
    return sympy_expr

def tokenize(prefix_expr:list, word2id:dict) -> list:
    tokenized_expr = []
    tokenized_expr.append(word2id["S"])
    for i in prefix_expr:
        tokenized_expr.append(word2id[i])
    tokenized_expr.append(word2id["F"])
    return tokenized_expr

def de_tokenize(tokenized_expr, id2word:dict):
    prefix_expr = []
    for i in tokenized_expr:
        if "F" == id2word[i]:
            break
        else:
            prefix_expr.append(id2word[i])
    return prefix_expr

def tokens_padding(tokens):
    max_len = max([len(y) for y in tokens])
    p_tokens = torch.zeros(len(tokens), max_len)
    for i, y in enumerate(tokens):
        p_tokens[i, :] = torch.cat([y.long(), torch.zeros(max_len - y.shape[0]).long()])
    return p_tokens


def evaluate_and_wrap(eqs: List[Equation]):
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


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        cfg: Params
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        #self.env_path = env_path
        self.batch = cfg.batch_size
        self.datamodule_params_train = cfg.datamodule_params_train
        self.datamodule_params_val = cfg.datamodule_params_val
        self.num_of_workers = cfg.num_of_workers

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            data_train = load_data(self.train_path)
            data_val = load_data(self.val_path)
            self.training_dataset = NesymresDataset(
                data_train,
                self.datamodule_params_train,
            )
            self.validation_dataset = NesymresDataset(
                data_val,
                self.datamodule_params_val,
            )
            if self.test_path:
                data_test = load_data(self.test_path)
                self.test_dataset = NesymresDataset(
                    data_test, self.datamodule_params_train
                )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
            num_workers=self.num_of_workers,
            pin_memory=True
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_of_workers,
            pin_memory=True
        )
        # if self.test_path:
        # testloader = torch.utils.data.DataLoader(
        #     self.test_dataset,
        #     batch_size=self.batch,
        #     shuffle=False,
        #     collate_fn=custom_collate_fn,
        #     num_workers=self.num_of_workers,
        #     pin_memory=True
        # )
        return validloader