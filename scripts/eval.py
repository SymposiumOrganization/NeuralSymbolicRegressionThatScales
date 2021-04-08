import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from typing import Tuple
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.dclasses import Params, DataModuleParams, FitParams, ArchitectureParams
from nesymres.utils import load_metadata_hdf5
from functools import partial
import hydra

@hydra.main(config_name="fitfunc")
def main(cfg):
    model_path = "/home/gem/repos/NeuralSymbolicRegressionThatScales/outputs/2021-04-07/07-09-01/Exp_weights/data/datasets/20M/20M_train_hdfs_log_-epoch=00-val_loss=5.96.ckpt"
    test_data = load_metadata_hdf5(hydra.utils.to_absolute_path(cfg.test_path))
    data_params = Params(datamodule_params_test=DataModuleParams(
                                total_variables=list(test_data.total_variables), 
                                total_coefficients=list(test_data.total_coefficients)),num_of_workers=0)

    architecture_params = ArchitectureParams()
    params_fit = FitParams(word2id=test_data.word2id, 
                            id2word=test_data.id2word, 
                            una_ops=test_data.una_ops, 
                            bin_ops=test_data.bin_ops, 
                            total_variables=list(test_data.total_variables),  
                            total_coefficients=list(test_data.total_coefficients),
                            rewrite_functions=list(test_data.rewrite_functions)
                            )
    data = DataModule(
        None,
        None,
        hydra.utils.to_absolute_path(cfg.test_path),
        cfg=data_params
    )
    data.setup()
    model = Model.load_from_checkpoint(model_path, cfg=architecture_params)
    model.eval()
    model.cuda()
    fitfunc = partial(model.fitfunc,cfg_params=params_fit)

    for i in data.test_dataloader():
        X,y, expr = i[0][:,:-1], i[0][:,-1:], i[2]
        breakpoint()
        # output = fitfunc(X,y)        
        # breakpoint()

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # ,1,2,4,5,6,7" Change Me
    #print(f"Starting a run with {config}")
    main()