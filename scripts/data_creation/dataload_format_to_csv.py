import pandas as pd 
import numpy as np
import multiprocessing
from multiprocessing import Manager
import click
import warnings
from tqdm import tqdm
import json
import os
from nesymres.dataset import generator
import time
import signal
from nesymres import dclasses
from pathlib import Path
import pickle
from sympy import lambdify
from nesymres.utils import create_env, load_dataset
from nesymres.dataset import data_utils 
import copyreg
import types
from itertools import chain
import traceback
import sympy as sp
from nesymres.dataset.sympy_utils import add_multiplicative_constants, add_additive_constants
import random
import hydra


@hydra.main(configconfig_name="../config")
def converter(cfg):
    # csv_availables = os.listdir(folder_csv)
    # for file_csv in csv_availables:
    df = pd.DataFrame()
    dataset = load_dataset(hydra.utils.to_absolute_path(cfg.test_path))
    for eq in dataset.eqs:
        cfg_const = cfg.dataset_test.constants
        dict_const = data_utils.sample_constants(eq,cfg_const)
        # num_constant = random.randint(0,
        breakpoint()
        df 
    
    for i in range(4):
        path_csv = os.path.join(folder_csv,file_csv)
        validation = pd.read_csv(path_csv)
        copyreg.pickle(types.CodeType, code_pickler, code_unpickler) #Needed for serializing code objects
        env, config_dict = create_env("config.json")
        env_pip = Pipepile(env, is_timer=False)

        res = []
        for idx in range(len(validation)):
            gt_expr = validation.iloc[idx]["gt_expr"]
            gt_expr = gt_expr.replace("pow","Pow")
            variables = list(eval(validation.iloc[idx]["support"]).keys())
            support = validation.iloc[idx]["support"]
            curr = env_pip.convert_lambda(gt_expr,variables,support) 
            res.append(curr)
        
        

        dataset = dclasses.Dataset(eqs=res, 
                                config=config_dict, 
                                total_variables=list(env.variables), 
                                total_coefficients=env.coefficients, 
                                word2id=env.word2id, 
                                id2word=env.id2word,
                                una_ops=env.una_ops,
                                bin_ops=env.una_ops,
                                rewrite_functions=env.rewrite_functions)
            
        Path("data/benchmarks").mkdir(parents=True, exist_ok=True)
        file_name = Path(file_csv).stem
        path = os.path.join("data/benchmarks", file_name)
        with open(path, "wb") as file:
            pickle.dump(dataset, file)

if __name__ == "__main__":
    converter()