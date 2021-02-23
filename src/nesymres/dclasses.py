from dataclasses import dataclass
from types import CodeType
from typing import List

@dataclass
class Equation:
    code: CodeType
    format_string: str
    symbols: set


@dataclass
class Dataset:
    eqs: List[Equation]
    config: dict



@dataclass
class DatasetParams:
    env_base_seed: int
    max_len: int
    same_nb_ops_per_batch: str
    export_data: str
    reload_data: str
    reload_size: int
    env_name: str
    operators: str
    max_ops: int
    int_base: int
    precision: int
    rewrite_functions: str
    leaf_probs: str
    variables: list
    n_coefficients: int
    beam_eval: bool
    beam_size: int
    beam_length_penalty: int
    beam_early_stopping: bool
    eval_only: bool
    n_words: int
    eos_index: int
    pad_index: int