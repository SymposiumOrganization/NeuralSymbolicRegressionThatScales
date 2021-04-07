from nesymres.utils import load_data, load_numpy_data_data
import h5py
import numpy as np
from io import BytesIO
import pickle
from nesymres.dclasses import Dataset
from tqdm import tqdm 
import multiprocessing
from pathlib import Path
import os
import click

class H5FilesCreator():
    def __init__(self,path,dataset):
        self.path = path

    def create_single_hd5(self,block):
        idx, eqs = block
        t_hf = h5py.File(os.path.join(self.path, str(idx) + ".h5") , 'w')
        for i, eq in enumerate(eqs):            
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()


def create_hdf_files(d: Dataset, s: str) -> None:
    num_eqs_per_set = int(1e5)
    n_datasets = int(len(d.eqs) // num_eqs_per_set) + 1
    path = Path(s+"_hdfs")
    h5_creator = H5FilesCreator(path, dataset=d)
    counter = 0
    path.mkdir(mode=0o777, parents=True, exist_ok=True)
    print("Diving sets")
    sets = [[d.eqs[i] for idx in range(i*num_eqs_per_set,(i+1)*num_eqs_per_set)] for i in range(n_datasets)]
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        max_ = n_datasets
        with tqdm(total=max_) as pbar:
            for f in p.imap_unordered(
                h5_creator.create_single_hd5, enumerate(sets)
            ):
                pbar.update()
    total_number_of_eqs = len(d.eqs)
    d.eqs = []
    d.total_number_of_eqs = total_number_of_eqs
    t_hf = h5py.File(os.path.join(path, "other" + ".h5") , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(d)))
    t_hf.close()



@click.command()
@click.option("--folder_dataset", default="data/datasets/20M/")
def main(folder_dataset):
    train_path = os.path.join(folder_dataset,"hdf_train")
    pathlib.Path(train_path).mkdir(parents=True, exist_ok=True) 
    val_path = os.path.join(folder_dataset,"hdf_val_subset")
    pathlib.Path(val_path).mkdir(parents=True, exist_ok=True) 
    print("Creating Validation h5")
    val_data = load_data(val_path)
    print("Validation Data Loaded, starting h5 creation")
    create_hdf_files(val_data, val_path)
    print("Creating Training h5")
    train_data = load_data(train_path)
    print("Training Data Loaded, starting h5 creation")
    create_hdf_files(train_data, train_path)
    Path.touch(mode=0o666, exist_ok=True)

if __name__=="__main__":
    main()