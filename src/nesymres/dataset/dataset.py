import numpy as np
import sympy
import warnings
import torch
from torch.utils import data
import math
from sympy_utils import check_additive_constants
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


def print_sympy(sym_expr):
    print(sym_expr.args)
    for i in sym_expr.args:
        print(type(i))
        print_sympy(i)
    print()


@dataclass
class DataPoint:
    df: pd.DataFrame
    train_index = None
    val_index = None


@dataclass
class InputBenchmarking:
    interpolation_pts: np.array
    extrapolation_pts: np.array
    gt_tokens: List


class wrapper_dataset(data.Dataset):
    def __init__(
        self,
        data,
        env,
        data_params,
    ):  # , transforms=None):
        # df = data.df.reset_index(drop=True)
        # self.df = df
        m = Manager()
        #self.symbols = np.array(data["Symbol"],dtype=object)
        self.fun = np.array(list(
            map(
                lambda x: types.FunctionType(x, globals=globals(), name="f"),
                data["Funcs"],
            )
        ))
        #self.fun = np.array(data["Funcs"])

        #
        self.tokenized = np.array(data["Tokenized"], dtype=object)
        self.symbols = np.array(data["Symbol"],dtype=object)
        
        self.tokenized = m.dict({i:tok for i, tok in enumerate(data["Tokenized"])})
        self.symbols = m.dict({i:sy for i, sy in enumerate(data["Symbol"])})
        self.fun = m.dict({i:fun for i, fun in enumerate(data["Funcs"])})
        #self.expression = np.array(data["Expression"])
        #self.expression = m.dict({i:expr for i, expr in enumerate(data["Expression"])})
        if "Format_Expression" in data:
            self.expr_wit_c = np.array(data["Format_Expression"])
            #self.expr_wit_c = m.dict({i:tok for i, tok in enumerate(data["Format_Expression"])})
        else:
            self.expr_wit_c = None
        self.env = env
        self.type_of_sampling_points = np.array(data_params.type_of_sampling_points)
        self.p = np.array([data_params.max_number_of_points] * len(self.fun))
        self.c_dof = np.array(data_params.constant_degree_of_freedom)
        self.predict_c = np.array(data_params.predict_c)
        self.support_extremes=np.array(data_params.support_extremes)
        self.input_normalization = np.array(data_params.input_normalization)
        #self.input_normalization =data_params.e
        if "Support" in data.keys():
            self.support = np.array([
                [
                    Uniform(
                        s[0], s[1]
                    )
                    for s in eqs.values()
                ]
                for eqs in data["Support"]
            ],dtype=object)
        else:
            self.support = None

    def __getitem__(self, index):
        total_consts_symbols = self.get_constants_interval()
        f = self.fun[index]
        f = types.FunctionType(f, globals=globals(), name="f")
        symbols = self.symbols[index]
        tokens = torch.tensor(self.tokenized[index])
        if type(self.expr_wit_c) != type(None):
            format_string = self.expr_wit_c[index]
            n_const = self.env.count_number_of_constants(format_string)
            used_consts = random.randint(0, min(self.c_dof, n_const))
            available_constans_symbols = list(total_consts_symbols.keys())[:n_const]
            symbols_used = random.sample(available_constans_symbols, used_consts)
            for si in symbols_used:
                total_consts_symbols[si] = Uniform(1, 5)
            costs = [
                random.random()
                if not bool(total_consts_symbols[x].low == total_consts_symbols[x].high)
                else int(total_consts_symbols[x].low)
                for x in available_constans_symbols
            ]
            if self.predict_c:
                example = format_string.format(*tuple(costs))
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
        else:
            return (
                f,
                symbols,
                tokens,
                self.p[index],
                self.type_of_sampling_points,
                self.support[index],
                total_consts_symbols,
            )

    def get_constants_interval(self):
        avail_consts_symbols = {
            "c1": Uniform(1, 1),
            "c2": Uniform(1, 1),
            "c3": Uniform(1, 1),
            "c4": Uniform(1, 1),
            "c5": Uniform(1, 1),
            "c6": Uniform(1, 1),
            "c7": Uniform(1, 1),
            "c8": Uniform(1, 1),
            "c9": Uniform(1, 1),
            "c10": Uniform(1, 1),
        }
        return avail_consts_symbols

    def __len__(self):
        return len(self.fun)
