"""Simple script that checks that no test equation are in the training dataset"""
from nesymres.utils import load_eq, load_metadata_hdf5
import click
import pandas as pd
from torch.distributions.uniform import Uniform
from nesymres.dataset.data_utils import create_uniform_support, sample_symbolic_constants
from nesymres.dataset.data_utils import evaluate_fun
import warnings
from sympy import lambdify,sympify, simplify
import multiprocessing
import torch
from tqdm import tqdm
import numpy as np
import os


def evaluate_validation_set(validation_eqs: pd.DataFrame, support) -> set:
    res = set()
    for _, row in validation_eqs.iterrows():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            variables = [f"x_{i}" for i in range(1,1+support.shape[0])]
            curr = tuple(lambdify(variables,row["eq"])(*support).numpy().astype('float16'))
            res.add(curr)
    return res


class Pipeline:
    """"""
    def __init__(self, data_path, metadata, support, target_image, validation_eqs: set):
        """
        Args:
            param1: 
            param2: 
            support: 
            target_image: 
            validation_eqs: A set containing all the validation equations in a str format and without constant placeholders. 
                            This argument is used for the symbol checking
        """
        self.data_path = data_path
        self.metadata = metadata
        self.support = support
        self.target_image = target_image
        self.validation_eqs = validation_eqs

    def is_valid_and_not_in_validation_set(self, idx: int) -> bool:
        """
        Assert both symbolically and numerically that the equation is not in the validation set
        Args:
            idx: index to the Eq in the dataset
        """
        # consts = torch.stack([torch.ones([int(self.support.shape[1])]) for i in self.metadata.total_coefficients])
        # input_lambdi = torch.cat([self.support,consts],axis=0)
        # assert input_lambdi.shape[0]  == len(self.metadata.total_coefficients) + len(self.metadata.total_variables)
        eq = load_eq(self.data_path, idx, self.metadata.eqs_per_hdf)
        # variables = [f"x_{i}" for i in range(1,1+self.support.shape[0])]
        # consts = [c for c in self.metadata.total_coefficients]
        #symbols = variables + consts
        
        #Symbol Checking
        const, dummy_const = sample_symbolic_constants(eq)
        eq_str = str(simplify(sympify(eq.expr.format(**dummy_const))))
        if eq_str in self.validation_eqs:
            print(idx)
            print(eq_str)
        #assert not eq_str in self.validation_eqs
        
        #Numerical Checking
        # args = [ eq.code,input_lambdi ]
        # y = evaluate_fun(args)
        # val = tuple(y)
        
        # if (val in self.target_image):
        #     print("Validation set {}:".format(list(self.target_image).index(val)))
        #     print("Equation in dataset {}".format(idx))
        #     print("Found in validation")
        #     return idx, False
        # return idx, True

        

@click.command()
@click.option("--data_path", default="data/datasets/10000000/")
@click.option("--csv_path", default="data/benchmark/nc_old.csv")
@click.option("--debug/--no-debug", default=False)
def main(data_path,csv_path,debug):
    metatada = load_metadata_hdf5(data_path)
    validation = pd.read_csv(csv_path)
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    target_image = evaluate_validation_set(validation,support)
    pipe = Pipeline(data_path, metatada, support, target_image,  set(validation["eq"]))
    total_eq = int(metatada.total_number_of_eqs)
    res = []
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            chunksize = 10000
            print("Chunksize of {}. Progress bar will update with this resolution".format(chunksize))
            with tqdm(total=total_eq) as pbar:
                for evaled in p.imap_unordered(pipe.is_valid_and_not_in_validation_set, list(range(total_eq)),chunksize=chunksize):
                    pbar.update()
                    res.append(evaled)
    else:
        res = list(map(pipe.is_valid_and_not_in_validation_set, tqdm(range(total_eq))))


if __name__=="__main__":
    main()