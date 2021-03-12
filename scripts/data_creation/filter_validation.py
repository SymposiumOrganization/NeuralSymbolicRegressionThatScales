import numpy as np
from nesymres.utils import code_unpickler, code_pickler, load_data
import click
import pickle
from nesymres import dclasses

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
    eqs_unique = [x for idx, x in enumerate(data_val.eqs) if idx in data_val.unique_index]
    to_keep = choose_eqs(eqs_unique)[:num_target_eq]
    validation_path = val_path + "_subset_{}".format(num_target_eq)
    with open(validation_path, "wb") as file:
        pickle.dump(subset, file)




if __name__=="__main__":
    main()