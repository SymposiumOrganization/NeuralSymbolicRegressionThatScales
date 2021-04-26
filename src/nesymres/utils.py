import marshal
import copyreg
import types
import pickle
import json
from .dataset import generator
from .dclasses import DatasetDetails, Equation, GeneratorDetails
from typing import List, Tuple
import h5py
import os
import numpy as np
from pathlib import Path

class H5FilesCreator():
    def __init__(self,path: Path):
        if int(path.stem) > 1e6:
            file_name = path.parent / "{}M".format(int(path.stem) // 1e6)
        elif int(path.stem) > 1e3:
            file_name = path.parent / "{}K".format(int(path.stem) // 1e3)
        path.mkdir(mode=0o777, parents=True, exist_ok=True)
        self.path = path
        

    def create_single_hd5(self,idx:int, eqs: List[Equation]):
        #idx, eqs = block
        t_hf = h5py.File(os.path.join(self.path, str(idx) + ".h5") , 'w')
        for i, eq in enumerate(eqs):            
            curr = np.void(pickle.dumps(eq))
            t_hf.create_dataset(str(i), data=curr)
        t_hf.close()


def code_unpickler(data):
    return marshal.loads(data)

def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)

# def load_numpy_data_data(f) -> Dataset:
#     copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
#     data = pickle.load(f)
#     return data

# def load_dataset(path_dataset) -> Dataset:
#     copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
#     with open(path_dataset, "rb") as f:
#         data = pickle.load(f)
#     return data


def load_eq(path_folder, idx, num_eqs_per_set) -> Equation:
    index_file = str(int(idx/num_eqs_per_set))
    f = h5py.File(os.path.join(path_folder,f"{index_file}.h5"), 'r')
    dataset_metadata = f[str(idx - int(index_file)*int(num_eqs_per_set))]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    f.close()
    return metadata

def load_metadata_hdf5(path_folder) -> DatasetDetails:
    f = h5py.File(os.path.join(path_folder,"metadata.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def create_env(path)->Tuple[generator.Generator,GeneratorDetails]:
    with open(path) as f:
        d = json.load(f)
    param = GeneratorDetails(**d)
    env = generator.Generator(param)
    return env, param, d

