import marshal
import copyreg
import types
import pickle
import json
from . import dclasses
from .dataset import generator
from .dclasses import Dataset

def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)


def load_data(path_dataset) -> Dataset:
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    with open(path_dataset, "rb") as f:
        data = pickle.load(f)
    return data

def create_env(path):
    with open(path) as f:
        d = json.load(f)
    param = dclasses.DatasetParams(**d)
    env = generator.Generator(param)
    return env, d



