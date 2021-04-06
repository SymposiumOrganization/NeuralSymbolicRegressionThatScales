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
    d.eqs = []
    t_hf = h5py.File(os.path.join(path, "other" + ".h5") , 'w')
    t_hf.create_dataset("other", data=np.void(pickle.dumps(d)))
    t_hf.close()

def main():
    train_path = "data/datasets/20M/20M_train"
    val_path = "data/datasets/20M/20M_val_subset"
    print("Creating Validation h5")
    val_data = load_data(val_path)
    print("Validation Data Loaded, starting h5 creation")
    create_hdf_files(val_data, val_path)
    print("Creating Training h5")
    train_data = load_data(train_path)
    print("Training Data Loaded, starting h5 creation")
    create_hdf_files(train_data, train_path)

if __name__=="__main__":
    main()