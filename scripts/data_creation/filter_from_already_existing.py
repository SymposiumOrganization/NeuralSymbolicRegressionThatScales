import click
import numpy as np
from nesymres.utils import code_unpickler, code_pickler, load_data
import pandas as pd
from collections import defaultdict
from nesymres.dataset.data_utils import evaluate_fun, group_symbolically_indetical_eqs, evaluate_validation_set,create_uniform_support
from torch.distributions.uniform import Uniform
import torch
from nesymres import dclasses
import multiprocessing
from tqdm import tqdm
import os
from pathlib import Path
import pickle



def return_numerically_different_from_validation(data,disjoint_sets, validation_set=None):
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    support = create_uniform_support(sampling_distribution, len(data.total_variables), num_p)
    consts = torch.stack([torch.ones([int(num_p)]) for i in data.total_coefficients])
    input_lambdi = torch.cat([support,consts],axis=0)
    assert input_lambdi.shape[0]  == len(data.total_coefficients) + len(data.total_variables)
    fun = [data.eqs[x[0]].code if x else [] for x in disjoint_sets]
    cut_off = min(int(5e4), len(data.eqs))
    points_fin, eqs_fin, lambi_fin = [], [], []
    res = []
    validation_set = evaluate_validation_set(validation_set, support)
    for i in range(int(len(data.eqs)/cut_off)+1):
        args = list(zip(fun[cut_off*i:cut_off*(i+1)], [input_lambdi]*cut_off))
        with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                with tqdm(total=cut_off) as pbar:
                    for evaled in p.map(evaluate_fun, args,chunksize=1000):
                            pbar.update()
                            res.append(evaled)

    disjoint_set_updated = [[] for _ in range(len(data.eqs))]
    seen = defaultdict(list)
    counter = 0
    for i, val in enumerate(res):
        if not len(val):
            continue
        val = tuple(val)
        assert tuple(val) == tuple(val)
        if val in validation_set:
            counter = counter + 1
            continue
        else:
            disjoint_set_updated[i].extend(disjoint_sets[i])
            

    print("Equations in test set".format(counter/len(data.eqs)))
    return disjoint_set_updated


@click.command()
@click.option("--data_path", default=None)
@click.option("--csv_path", prompt='Insert the csv path', default="data/csv/ours_all_flavours.csv")
def main(data_path,csv_path):
    print("Started Loading Data")
    data = load_data(data_path)
    validation = pd.read_csv(csv_path)
    print("Loading Data Complete")

    print("Grouping equations with identical symbolic form")
    indexes_dict = defaultdict(list)
    disjoint_sets = [[] for _ in range(len(data.eqs))]
    indexes_dict, disjoint_sets = group_symbolically_indetical_eqs(data, indexes_dict, disjoint_sets)

    #validation_set = set(validation.loc[validation["benchmark"]=="ours-nc"]["gt_expr"])
    disjoint_set_updated = return_numerically_different_from_validation(data,disjoint_sets, validation_set=set(validation["gt_expr"]))

    print("Creating Training Set")
    training_indices = []
    for i in disjoint_set_updated:
        training_indices.extend(i)
    train_eqs = [data.eqs[x] for x in training_indices] 
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