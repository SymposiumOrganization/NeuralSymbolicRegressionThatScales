import marshal
import copyreg
import types
import pickle
import json
from . import dclasses
from .dataset import generator
from .dclasses import Dataset
import h5py
import os
import numpy as np

def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)

def load_numpy_data_data(f) -> Dataset:
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    data = pickle.load(f)
    return data

def load_data(path_dataset) -> Dataset:
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    with open(path_dataset, "rb") as f:
        data = pickle.load(f)
    return data

def load_eqs(path_folder, eq ):
    f = h5py.File(os.path.join(path_folder,"other.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def load_metadata_hdf5(path_folder) -> Dataset:
    f = h5py.File(os.path.join(path_folder,"other.h5"), 'r')
    dataset_metadata = f["other"]
    raw_metadata = np.array(dataset_metadata)
    metadata = pickle.loads(raw_metadata.tobytes())
    return metadata

def create_env(path):
    with open(path) as f:
        d = json.load(f)
    param = dclasses.DatasetParams(**d)
    env = generator.Generator(param)
    return env, d



