from dataclasses import dataclass
from types import CodeType
from typing import List, Tuple
from torch.distributions import Uniform, Normal, Distribution
from dataclass_dict_convert import dataclass_dict_convert
import torch

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
class NNEquation:
    numerical_values: torch.tensor
    tokenized: torch.tensor
    expr: List[str]


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
    una_ops: list
    bin_ops: list
    rewrite_functions: list 
    unique_index: set = None
    total_number_of_eqs: int = 0
    
    
    
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

# @dataclass
# class DataModuleParams:
#     #constant_options: ConstantsOptions
#     #max_number_of_points: int = 500
#     #type_of_sampling_points: str = ["constant", "logarithm"][1]
#     #predict_c: bool = False
#     # fun_support: tuple = (-10,10)
#     # distribution_support: Distribution = [Uniform, Normal][0]
#     total_variables: list = None
#     total_coefficients: list = None
    
@dataclass_dict_convert()
@dataclass(frozen=True)
class ArchitectureParams:
    sinuisodal_embeddings: bool = False
    dec_pf_dim: int = 512
    dec_layers: int = 2
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
    length_eq: int = 60
    n_l_enc: int = 2
    mean: float = 0.5  
    std: float = 0.5 
    dim_input: int = 4
    num_heads: int = 8
    output_dim: int = 60
    dropout: float = 0



# @dataclass
# class Params:
#     datamodule_params_train: DataModuleParams = DataModuleParams()
#     datamodule_params_val: DataModuleParams = DataModuleParams()
#     datamodule_params_test: DataModuleParams = DataModuleParams()
#     num_of_workers: int = 0
#     max_epochs: int = 200
#     val_check_interval: float = 0.2
#     precision: int = 16
#     batch_size: int = 100



@dataclass
class BFGSParams:
    on: bool = True
    n_restarts: bool = 10
    add_coefficients_if_not_existing: bool = True
    normalization_o: bool = False
    idx_remove: bool = True
    normalization_type: str = ["MSE","NMSE"][0]
    stop_time: int = 1e9

@dataclass
class FitParams:
    word2id: dict
    id2word: dict
    total_coefficients: list
    total_variables: list
    rewrite_functions: list
    una_ops: list = None
    bin_ops: list = None
    bfgs: BFGSParams = BFGSParams()
    beam_size: int = 3
    
# @dataclass
# class ConstantsOptions:
#     max_constants: int
#     min_additive_constant_support: float
#     max_additive_constant_support: float 
#     min_multiplicative_constant_support: float
#     max_multiplicative_constant_support: float