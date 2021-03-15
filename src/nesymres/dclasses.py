from dataclasses import dataclass
from types import CodeType
from typing import List, Tuple

@dataclass
class Equation:
    code: CodeType
    expr: str
    coeff_dict: dict
    variables: set
    support: tuple = None
    tokenized: list = None
    valid: bool = True

@dataclass
class DataModuleParams:
    max_number_of_points: int
    type_of_sampling_points: str
    support_extremes: Tuple
    constant_degree_of_freedom: int
    predict_c: bool
    distribution_support: str
    input_normalization: bool


@dataclass
class Dataset:
    eqs: List[Equation]
    config: dict
    total_coefficients: list
    total_variables: list
    word2id: dict
    id2word: dict
    unique_index: set = None
    
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

@dataclass
class DataModuleParams:
    max_number_of_points: int = 500
    type_of_sampling_points: str = ["Constant", "Logarithm"][1]
    predict_c: bool = False
    constant_degree_of_freedom: int = 3
    fun_support: tuple = (-10,10)
    constant_support: tuple = (-10,10)
    distribution_support: str = ["Uniform", "Gaussian"][0]

@dataclass
class Architecture:
    sinuisodal_embeddings: bool = False
    dec_pf_dim: int = 512
    dec_layers: int = 5 
    dim_hidden: int = 512  
    lr: int = 0.0001
    dropout: int = 0
    num_features: int = 10
    ln: bool = True
    N_p:int = 0
    num_inds: int = 50
    activation: str = "relu"
    bit16: bool = True
    norm: bool = True
    linear: bool= False
    input_normalization: bool = False
    src_pad_idx: int = 0
    trg_pad_idx: int = 0
    length_eq: int = 30
    n_l_enc: int = 4
    mean: float = 0.5  
    std: float = 0.5 
    dim_input: int = 4
    num_heads: int = 8
    output_dim: int = 33
    dropout: float = 0



@dataclass
class Params:
    datamodule_params_train: DataModuleParams = DataModuleParams()
    datamodule_params_val: DataModuleParams = DataModuleParams()
    datamodule_params_test: DataModuleParams = DataModuleParams()
    architecture: Architecture = Architecture()
    num_of_workers: int = 0
    max_epochs: int = 200
    val_check_interval: float = 0.05
    precision: int = 16
    batch_size: int = 450

