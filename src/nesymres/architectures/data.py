import numpy as np
import sympy
import warnings
import torch
from torch.utils import data
import math
import pandas as pd
from dataclasses import dataclass
from sympy.core.rules import Transform
from sympy import sympify, Symbol
from sympy import trigsimp
from sympy import Float
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

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import pytorch_lightning as pl
from ..utils import load_data, create_env
from nesymres.dclasses import Params

class NesymresDataset(data.Dataset):
    def __init__(
        self,
        data,
        env,
        data_params: DataModuleParams,
    ):  
        m = Manager()
        self.eqs = m.dict({i:eq for i, eq in enumerate(data.eqs)})
        self.env = env
        self.data_params = data_params

    def __getitem__(self, index):
        eq = self.eqs[index]
        code = types.FunctionType(eq.code, globals=globals(), name="f")
        consts = {const: 1 if const[:2] == "cm" else 0 for const in eq.coeff_dict.keys()}
        used_consts = random.randint(0, min(len(eq.coeff_dict),self.data_params.constant_degree_of_freedom))
        symbols_used = random.sample(set(eq.coeff_dict.keys()), used_consts)
        for si in symbols_used:
            consts[si] = Uniform(self.data_params.constant_support[0], self.data_params.constant_support[1]).sample()
        
        # costs = [
        #     random.random()
        #     if not bool(total_consts_symbols[x].low == total_consts_symbols[x].high)
        #     else int(total_consts_symbols[x].low)
        #     for x in available_constans_symbols
        # ]
        if self.data_params.predict_c:
            example = eq.expr.format(**consts)
            sympy_expr = self.env.constants_to_placeholder(example)
            if self.input_normalization:
                sympy_expr = self.env.constants_to_placeholder(example)*Symbol('c') + Symbol('c')
            #sympy_expr = trigsimp(sympy_expr)  # Delete if slow
            try:
                prefix = self.env.sympy_to_prefix(sympy_expr)
                t = self.env.tokenize(prefix)
                tokens = torch.tensor(t)
            except:
                #print("Error with {}".format(sympy_expr))  # lUCA COMMENT
                tokens = torch.tensor([0, 0, 0, 0, 0])
                # breakpoint()
        support = []
        min_supp_len = 2
        for i in range(len(symbols)):
            mi = np.random.uniform(self.support_extremes[0], self.support_extremes[1]-min_supp_len)
            ma = np.random.uniform(mi+min_supp_len, self.support_extremes[1])
            support.append(Uniform(mi, ma))
        return (
            f,
            symbols,
            tokens,
            self.p[index],
            self.type_of_sampling_points,
            support,
            total_consts_symbols,
        )
    # else:
    #     return (
    #         f,
    #         symbols,
    #         tokens,
    #         self.p[index],
    #         self.type_of_sampling_points,
    #         self.support[index],
    #         total_consts_symbols,
    #     )



    def __len__(self):
        return len(self.eqs)



def custom_collate_fn(y):
    (
        function,
        symbols,
        tokens,
        p,
        type_of_sampling_points,
        support,
        constants_interval,
    ) = (
        list(zip(*y))[0],
        list(zip(*y))[1],
        list(zip(*y))[2],
        list(zip(*y))[3],
        list(zip(*y))[4],
        list(zip(*y))[5],
        list(zip(*y))[6],
    )
    tokens = tokens_padding(tokens)
    # p = random.randrange(500,1500)
    res, tokens = evaluate_and_wrap(
        function,
        symbols,
        tokens,
        p,
        type_of_sampling_points,
        support,
        constants_interval,
    )


    return res, tokens




class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        env_path,
        train_path,
        val_path,
        test_path,
        cfg: Params
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.env_path = env_path
        self.batch = cfg.batch_size
        self.datamodule_params_train = cfg.datamodule_params_train
        self.datamodule_params_val = cfg.datamodule_params_val
        self.num_of_workers = cfg.num_of_workers

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            data_train = load_data(self.train_path)
            env,_ = create_env(self.env_path)
            data_val = load_data(self.val_path)
            self.training_dataset = NesymresDataset(
                data_train,
                env,
                self.datamodule_params_train,
            )
            self.validation_dataset = NesymresDataset(
                data_val,
                env,
                self.datamodule_params_val,
            )
            if self.test_path:
                data_test = load_data(self.test_path)
                self.test_dataset = NesymresDataset(
                    data_test, env, self.datamodule_params_train
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