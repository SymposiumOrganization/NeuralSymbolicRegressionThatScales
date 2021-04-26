import click
import numpy as np
from nesymres.utils import code_unpickler, code_pickler, load_eq, load_metadata_hdf5
import pandas as pd
from collections import defaultdict
from nesymres.dataset.data_utils import create_uniform_support
from nesymres.benchmark import evaluate_func, return_order_variables
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


def evaluate_validation_set(validation_eqs, support) -> set:
    res = set()
    for i in validation_eqs:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            curr = tuple(lambdify(['x_1','x_2','x_3'],i)(*support).numpy().astype('float16'))
            res.add(curr)
    return res

# def return_numerically_different_from_validation(data,disjoint_sets, validation_set=None):
#     sampling_distribution = Uniform(-25,25)
#     num_p = 400
#     support = create_uniform_support(sampling_distribution, len(data.total_variables), num_p)
#     consts = torch.stack([torch.ones([int(num_p)]) for i in data.total_coefficients])
#     input_lambdi = torch.cat([support,consts],axis=0)
#     assert input_lambdi.shape[0]  == len(data.total_coefficients) + len(data.total_variables)
#     fun = [data.eqs[x[0]].code if x else [] for x in disjoint_sets]
#     cut_off = min(int(5e4), len(data.eqs))
#     points_fin, eqs_fin, lambi_fin = [], [], []
#     res = []
#     validation_set = evaluate_validation_set(validation_set, support)
#     for i in range(int(len(data.eqs)/cut_off)+1):
#         args = list(zip(fun[cut_off*i:cut_off*(i+1)], [input_lambdi]*cut_off))
#         with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
#                 with tqdm(total=cut_off) as pbar:
#                     for evaled in p.map(evaluate_fun, args,chunksize=1000):
#                             pbar.update()
#                             res.append(evaled)

#     disjoint_set_updated = [[] for _ in range(len(data.eqs))]
#     seen = defaultdict(list)
#     counter = 0
#     for i, val in enumerate(res):
#         if not len(val):
#             continue
#         val = tuple(val)
#         assert tuple(val) == tuple(val)
#         if val in validation_set:
#             counter = counter + 1
#             continue
#         else:
#             disjoint_set_updated[i].extend(disjoint_sets[i])
            

#     print("Equations in test set".format(counter/len(data.eqs)))
#     return disjoint_set_updated

class Pipeline:
    def __init__(self, data_path, metadata, support, target_image):
        self.data_path = data_path
        self.metadata = metadata
        self.support = support
        self.target_image = target_image

    def is_not_eq_in_validation_or_always_zero(self, index) -> bool:
        sampling_distribution = Uniform(-25,25)
        num_p = 400
        support = create_uniform_support(sampling_distribution, len(metadata.total_variables), num_p)
        consts = torch.stack([torch.ones([int(num_p)]) for i in metadata.total_coefficients])
        input_lambdi = torch.cat([support,consts],axis=0)
        assert input_lambdi.shape[0]  == len(metadata.total_coefficients) + len(metadata.total_variables)
        eq = load_eq(self.data_path, index, metadata.eqs_per_hdf)
        y = evaluate_fun(eq.code, input_lambdi)


@click.command()
@click.option("--data_path", default="data/raw_datasets/10M/")
@click.option("--csv_path", default="data/benchmark/nc_old.csv")
def main(data_path,csv_path):
    print("Loading metadata")
    metatada = load_metadata_hdf5(data_path)
    #data = load_dataset(data_path)
    validation = pd.read_csv(csv_path)
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(metatada.total_variables), num_p)
    print("Creating image for validation set")
    target_image = evaluate_validation_set(validation["eq"],support)
    pipe = Pipeline(data_path, metatada, support, target_image)
    print("Starting finding out index equations in the validation set")
    if not debug:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
            with tqdm(total=metatada.total_number_of_eqs) as pbar:
                for evaled in p.map(pipe.is_not_eq_in_validation_or_always_zero, list(range(metatada.total_number_of_eqs)),chunksize=1000):
                    pbar.update()
                    res.append(evaled)
    else:
        for 


    train_excludes_set = dclasses.Dataset(eqs=train_eqs, 
                                        config=data.config, 
                                        total_coefficients=data.total_coefficients, 
                                        total_variables=data.total_variables, 
                                        word2id=data.word2id, 
                                        id2word=data.id2word,
                                        una_ops=data.una_ops,
                                        bin_ops=data.bin_ops,
                                        rewrite_functions=data.rewrite_functions)   
    
    p = os.path.join(Path(data_path).parent.parent, "datasets", Path(data_path).stem)
    Path(p).mkdir(parents=True, exist_ok=True)
    training_path = os.path.join(p,Path(p).stem + "_filtered")
    with open(training_path, "wb") as file:
        pickle.dump(train_excludes_set, file)

    

if __name__=="__main__":
    main()