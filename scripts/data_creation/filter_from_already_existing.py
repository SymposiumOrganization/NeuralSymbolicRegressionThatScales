import click
import numpy as np
from nesymres.utils import code_unpickler, code_pickler, load_eq, load_metadata_hdf5
from nesymres.dataset.data_utils import evaluate_fun
import pandas as pd
from collections import defaultdict
from nesymres.dataset.data_utils import create_uniform_support
from nesymres.benchmark import return_order_variables
from torch.distributions.uniform import Uniform
import torch
from nesymres import dclasses
import multiprocessing
from tqdm import tqdm
import os
from pathlib import Path
import pickle
import warnings
from sympy import lambdify


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
    def __init__(self, data_path, metadata, support, target_image):
        self.data_path = data_path
        self.metadata = metadata
        self.support = support
        self.target_image = target_image

    def is_valid_and_not_in_validation_set(self, index) -> bool:
        consts = torch.stack([torch.ones([int(self.support.shape[1])]) for i in self.metadata.total_coefficients])
        input_lambdi = torch.cat([self.support,consts],axis=0)
        assert input_lambdi.shape[0]  == len(self.metadata.total_coefficients) + len(self.metadata.total_variables)
        eq = load_eq(self.data_path, index, self.metadata.eqs_per_hdf)
        variables = [f"x_{i}" for i in range(1,1+self.support.shape[0])]
        consts = [c for c in self.metadata.total_coefficients]
        #symbols = variables + consts
        args = [ eq.code,input_lambdi ]
        y = evaluate_fun(args)
        val = tuple(y)
        if val == tuple([]):
            print("Not an equation")
            return index, False
        if val == tuple([float("-inf")]*input_lambdi.shape[-1]):
            print("Found all Inf")
            return index, False
        if val == tuple([float("+inf")]*input_lambdi.shape[-1]):
           print("Found all -Inf")
           return index, False
        if val == tuple([float(0)]*input_lambdi.shape[-1]):
            print("Found all zeros")
            return index, False
        if val == tuple([float("nan")]*input_lambdi.shape[-1]):
            print("Found all nans")
            return index, False
        if val in self.target_image:
            print("Found in validation")
            return index, False
        return index, True

        

@click.command()
@click.option("--data_path", default="data/raw_datasets/10M/")
@click.option("--csv_path", default="data/benchmark/nc_old.csv")
@click.option("--debug/--no-debug", default=True)
def main(data_path,csv_path,debug):
    print("Loading metadata")
    metatada = load_metadata_hdf5(data_path)
    #data = load_dataset(data_path)
    validation = pd.read_csv(csv_path)
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    target_image = evaluate_validation_set(validation,support)
    pipe = Pipeline(data_path, metatada, support, target_image)
    print("Starting finding out index equations in the validation set")
    total_eq = int(metatada.total_number_of_eqs)
    res = []
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=total_eq) as pbar:
                for evaled in p.imap_unordered(pipe.is_valid_and_not_in_validation_set, list(range(total_eq)),chunksize=10000):
                    pbar.update()
                    res.append(evaled)
    else:
        res = list(map(pipe.is_valid_and_not_in_validation_set, tqdm(range(total_eq))))
    np.save(os.path.join(data_path,"filtered"),res)

if __name__=="__main__":
    main()