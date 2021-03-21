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
from nesymres.utils import create_env
from nesymres.utils import code_unpickler, code_pickler
import copyreg
import types
from itertools import chain
import traceback
import sympy as sp
from nesymres.dataset.sympy_utils import add_multiplicative_constants, add_additive_constants



class Pipepile:
    def __init__(self, env: generator.Generator, is_timer=False):
        self.env = env
        manager = Manager()
        self.cnt = manager.list()
        self.is_timer = is_timer
        self.fun_args = ",".join(chain(list(env.variables),env.coefficients))

    def handler(self,signum, frame):
        raise TimeoutError

    def return_training_set(self, i):
        np.random.seed(i)
        while True:
            try:
                res = self.create_lambda(np.random.randint(2**32-1))
                assert type(res) == dclasses.Equation
                return res
            except TimeoutError:
                signal.alarm(0)
                continue
            except generator.NotCorrectIndependentVariables:
                signal.alarm(0)
                continue
            except generator.UnknownSymPyOperator:
                signal.alarm(0)
                continue
            except generator.ValueErrorExpression:
                signal.alarm(0)
                continue
            except generator.ImAccomulationBounds:
                signal.alarm(0)
                continue

        

    def convert_lambda(self, i, variables):
        # if self.is_timer:
        #     signal.signal(signal.SIGALRM, self.handler)
        #     signal.alarm(1)
        sym = self.env.infix_to_sympy(i, self.env.variables, self.env.rewrite_functions)
        sym = sym.subs(sp.Symbol("x"), sp.Symbol("x1"))
        sym = sym.subs(sp.Symbol("y"), sp.Symbol("x2"))
        sym = sym.subs(sp.Symbol("z"), sp.Symbol("x3"))
        #pre = self.env.sympy_to_prefix(sym)
        # for idx,_ in enumerate(pre):
        #     if pre[idx] == "x":
        #         pre[idx] = "x1"
        #     elif pre[idx] == "y":
        #         pre[idx] = "x2"
        #     elif pre[idx] == "z":
        #         pre[idx] = "x3"
        placeholder = {x:sp.Symbol(x, real=True,nonzero=True) for x in ["cm","ca"]}
        sym = add_multiplicative_constants(sym, placeholder["cm"], unary_operators=self.env.una_ops)
        sym = add_additive_constants(sym,  placeholder, unary_operators=self.env.una_ops)
        pre = self.env.sympy_to_prefix(sym)
        prefix = self.env.add_identifier_constants(pre)
        consts =  self.env.return_constants(prefix)
        infix, _  = self.env._prefix_to_infix(prefix, coefficients=self.env.coefficients, variables=self.env.variables)
        consts_elemns = {y:y for x in consts.values() for y in x}
        constants_expression = infix.format(**consts_elemns)
        eq = lambdify(self.fun_args,constants_expression,modules=["numpy"])
        res = dclasses.Equation(expr=infix, code=eq.__code__, coeff_dict=consts_elemns, variables=variables)
        signal.alarm(0) 
        return res

@click.command()
@click.argument("path_csv")
def converter(path_csv):
    validation = pd.read_csv(path_csv)
    validation = validation.loc[validation["benchmark"]=="ours-nc"]
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler) #Needed for serializing code objects
    # total_number = number_of_equations
    # warnings.filterwarnings("error")
    env, config_dict = create_env("config.json")
    env_pip = Pipepile(env, is_timer=False)
    for eq in validation.iterrows():
        env_pip.convert_lambda(eq) 
    
    

    dataset = dclasses.Dataset(eqs=res, 
                               config=config_dict, 
                               total_variables=list(env.variables), 
                               total_coefficients=env.coefficients, 
                               word2id=env.word2id, 
                               id2word=env.id2word,
                               una_ops=env.una_ops,
                               bin_ops=env.una_ops,
                               rewrite_functions=env.rewrite_functions)
    print("Expression generation took {} seconds".format(time.time() - starttime))
    print(
        "Total number of equations created {}".format(
            len(res)
        )
    )
    
    if not debug:
        folder_path = "data/raw_datasets/" 
    else:
        folder_path = "data/raw_datasets/debug" 
        
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    file_name = "{}K".format(int(number_of_equations / 1000))
    path = os.path.join(folder_path, file_name)
    with open(path, "wb") as file:
        pickle.dump(dataset, file)

if __name__ == "__main__":
    converter()