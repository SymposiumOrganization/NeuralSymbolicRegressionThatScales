import numpy as np
from eq_learner.marshalling_files import code_unpickler, code_pickler, load_data
import click
from eq_learner.DatasetCreator.exploration import add_numerically_identically, create_symbolic_disjoint_set,create_subset
import pickle

def choose_eqs(res):
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
    exprs_val = data_val["Expression"]
    disjoint_sets = create_symbolic_disjoint_set(exprs_val)
    disjoint_sets, res = add_numerically_identically(data_val["Funcs"],disjoint_sets, extremes=(-10,10))
    to_keep = choose_eqs(res)[:num_target_eq]
    subset = create_subset(data_val, to_keep)
    validation_path = val_path + "_subset_{}".format(num_target_eq)
    with open(validation_path, "wb") as file:
        pickle.dump(subset, file)




if __name__=="__main__":
    main()