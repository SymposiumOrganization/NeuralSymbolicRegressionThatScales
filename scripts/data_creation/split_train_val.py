import marshal
import copyreg
import pickle
import json
import os
import click
import random
from collections import defaultdict
import numpy as np
from torch.distributions.uniform import Uniform
import torch
import multiprocessing
from tqdm import tqdm
import warnings
from nesymres.utils import code_unpickler, code_pickler, load_data
from nesymres import dclasses
from pathlib import Path
from nesymres.dataset.data_utils import evaluate_fun


def group_symbolically_indetical_eqs(data,indexes_dict,disjoint_sets):
    for i, val in enumerate(data.eqs):
        if not val.expr in indexes_dict:
            indexes_dict[val.expr].append(i)
            disjoint_sets[i].append(i)
        else:
            first_key = indexes_dict[val.expr][0]
            disjoint_sets[first_key].append(i)
    return indexes_dict, disjoint_sets

def group_numerical_indetical_eqs(data,disjoint_sets):
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    sym = {}
    for idx, sy in enumerate(data.total_variables):
        sym[idx] = sampling_distribution.sample([int(num_p)])
    consts = torch.stack([torch.ones([int(num_p)]) for i in data.total_coefficients])
    support = torch.stack([x for x in sym.values()])
    input_lambdi = torch.cat([support,consts],axis=0)
    assert input_lambdi.shape[0]  == len(data.total_coefficients) + len(data.total_variables)
    fun = [data.eqs[x[0]].code if x else [] for x in disjoint_sets]
    cut_off = min(int(5e4), len(data.eqs))
    points_fin, eqs_fin, lambi_fin = [], [], []
    res = []
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
        if type(val) == list and val == []:
            continue
        val = tuple(val)
        assert tuple(val) == tuple(val)
        if not val in seen:
            seen[val].append(i)
            disjoint_set_updated[i].extend(disjoint_sets[i])
        else:
            counter = counter + 1
            first_key = seen[val][0]
            disjoint_set_updated[first_key].extend(disjoint_sets[i])
    
    print("\% of numerically identical but different eqs {}".format(counter/len(data.eqs)))
    return disjoint_set_updated



@click.command()
@click.option(
    "--eqs_for_validation",
    default=10000,
    help="Number of equations for validation",
)
@click.option("--data_path", default=None)
def main(eqs_for_validation, data_path):
    print("Started Loading Data")
    data = load_data(data_path)
    if len(data.eqs) < eqs_for_validation:
        raise ("Not enough training equations")
    print("Finding Unique Expressions")
    seen = set()
    indexes_dict = defaultdict(list)
    disjoint_sets = [[] for _ in range(len(data.eqs))]
    indexes_dict, disjoint_sets = group_symbolically_indetical_eqs(data, indexes_dict, disjoint_sets)
    disjoint_sets = group_numerical_indetical_eqs(data, disjoint_sets)

    keys = []
    rep = []
    for k in disjoint_sets:
        keys.append(k)
        rep.append(len(k))

    keys = np.array(keys)
    rep = np.array(rep,dtype=np.float64)/np.array(len(data.eqs))
    print("Sampling Expressions for Validation")
    validation_indeces = []
    validation_entries = np.random.choice(keys,size=eqs_for_validation,replace=False,p=rep)

    for i in validation_entries:
        validation_indeces.extend(i)
    print("Creating Training Set")
    training_indeces = set(range(len(data.eqs)))- set(validation_indeces)
    assert not validation_indeces[0] in training_indeces

    train_eqs = [data.eqs[x] for x in training_indeces]
    training_dataset = dclasses.Dataset(eqs=train_eqs, config=data.config, total_coefficients=data.total_coefficients, total_variables=data.total_variables, word2id=data.word2id, id2word=data.id2word)
    val_eqs = [data.eqs[x] for x in validation_indeces]
    validation_dataset = dclasses.Dataset(eqs=val_eqs, config=data.config, total_coefficients=data.total_coefficients, total_variables=data.total_variables, word2id=data.word2id, id2word=data.id2word)

    t = [x.expr for x in training_dataset.eqs]
    t_n = [x for x in training_indeces if disjoint_sets[x]]
    v = [x.expr for x in validation_dataset.eqs]
    v_n = [x for x in validation_indeces if disjoint_sets[x]]
    print("Number in the training: ", len(training_dataset.eqs))
    print(f"Symbolically Unique number in the training: {len(set(t))},  {len(set(t))/len(training_dataset.eqs)} of training")
    print(f"Numerically Unique number in the training: {len(t_n)}, {len(set(t_n))/len(training_dataset.eqs)} of training")
    print("Number in the validation: ", len(validation_dataset.eqs))
    print(f"Symbolically Unique number in the validation: {len(set(v))}, {len(set(v))/len(validation_dataset.eqs)} of validation")
    print(f"Numerically Unique number in the validation: {len(v_n)}, {len(set(v_n))/len(validation_dataset.eqs)} of validation")
    
    assert len(v_n) == eqs_for_validation
    assert not len(
        set(t) & set(v)
    )
    
    train_unique_indices = [idx for idx, x in enumerate(training_indeces) if disjoint_sets[x]]
    training_dataset.unique_index = set(train_unique_indices)
    val_unique_indices = [idx for idx, x in enumerate(validation_indeces) if disjoint_sets[x]]
    validation_dataset.unique_index = set(val_unique_indices)

    p = os.path.join(Path(data_path).parent.parent, "datasets", Path(data_path).stem)
    Path(p).mkdir(parents=True, exist_ok=True)
    training_path = os.path.join(p,Path(p).stem + "_train")
    with open(training_path, "wb") as file:
        pickle.dump(training_dataset, file)

    validation_path= os.path.join(p,Path(p).stem + "_val")
    with open(validation_path, "wb") as file:
        pickle.dump(validation_dataset, file)

    Path(os.path.join(p, ".dirstamp")).touch()


if __name__ == "__main__":
    main()