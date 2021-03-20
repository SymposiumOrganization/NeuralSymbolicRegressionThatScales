import numpy as np
from nesymres.utils import code_unpickler, code_pickler, load_data
import click
import pickle
from nesymres import dclasses
from torch.distributions.uniform import Uniform
import torch
import multiprocessing
from tqdm import tqdm
from nesymres.dataset.data_utils import evaluate_fun


def choose_eqs(data):
    sampling_distribution = Uniform(-25,25)
    num_p = 400
    sym = {}
    for idx, sy in enumerate(data.total_variables):
        sym[idx] = sampling_distribution.sample([int(num_p)])
    consts = torch.stack([torch.ones([int(num_p)]) for i in data.total_coefficients])
    support = torch.stack([x for x in sym.values()])
    input_lambdi = torch.cat([support,consts],axis=0)
    assert input_lambdi.shape[0]  == len(data.total_coefficients) + len(data.total_variables)
    fun = [x.code for x in data.eqs]
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
    to_keep = []
    for idx, y in enumerate(res):
        if len(y) and min(y) != max(y) and max(y)< 1000 and min(y)>-1000:
            to_keep.append(idx)
    return to_keep

@click.command()
@click.option("--val_path")
@click.option("--num_target_eq", default=100)
def main(val_path,num_target_eq):
    print("Started Loading Data")
    data_val = load_data(val_path)
    data_val.eqs = [x for idx, x in enumerate(data_val.eqs) if idx in data_val.unique_index]
    to_keep = choose_eqs(data_val)[:num_target_eq]
    eqs = [data_val.eqs[x] for x in to_keep]
    filtered_validation = dclasses.Dataset(eqs=eqs, 
                                config=data_val.config, 
                                total_coefficients=data_val.total_coefficients, 
                                total_variables=data_val.total_variables, 
                                unique_index=range(len(to_keep)),  
                                word2id=data_val.word2id, 
                                id2word=data_val.id2word,
                                bin_ops=data_val.bin_ops,
                                una_ops=data_val.una_ops,
                                rewrite_functions=data_val.rewrite_functions)
    validation_path = val_path + "_subset"
    with open(validation_path, "wb") as file:
        pickle.dump(filtered_validation, file)




if __name__=="__main__":
    main()