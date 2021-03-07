from dataclasses import dataclass
from types import CodeType
from typing import List

@dataclass
class Equation:
    code: CodeType
    expr: str
    coeff_dict: dict
    variables: set



@dataclass
class Dataset:
    eqs: List[Equation]
    config: dict



@dataclass
class DatasetParams:
    max_len: int
    positive: bool
    env_name: str
    operators: str
    max_ops: int
    int_base: int
    precision: int
    rewrite_functions: str
    variables: list
    eos_index: int
    pad_index: int