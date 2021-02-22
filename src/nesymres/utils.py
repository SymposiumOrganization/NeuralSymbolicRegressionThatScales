import marshal
import copyreg
import types
import pickle


def code_unpickler(data):
    return marshal.loads(data)


def code_pickler(code):
    return code_unpickler, (marshal.dumps(code),)


def load_data(path_dataset):
    copyreg.pickle(types.CodeType, code_pickler, code_unpickler)
    with open(path_dataset, "rb") as f:
        data = pickle.load(f)
    return data

def create_env(d):
    param = Params(**d)
    env = generator.CharSPEnvironment(param)
    return env