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
from nesymres.dclasses import Params, DataModuleParams, FitParams, Architecture
from nesymres.utils import load_data
from functools import partial


def main():
    model_path = "Exp_weights/data/datasets/100K/100K_train_log_-epoch=06-val_loss=1.19.ckpt"
    test_data = load_data("data/datasets/100K/100K_val_subset")
    data_params = Params(datamodule_params_test=DataModuleParams(
                                total_variables=list(test_data.total_variables), 
                                total_coefficients=list(test_data.total_coefficients)),num_of_workers=0)

    architecture_params = Architecture()
    params_fit = FitParams(word2id=test_data.word2id, id2word=test_data.id2word)
    data = DataModule(
        None,
        None,
        test_data,
        cfg=data_params
    )
    data.setup()
    model = Model.load_from_checkpoint(model_path, cfg=architecture_params)
    model.eval()
    model.cuda()
    fitfunc = partial(model.fitfunc,cfg_params=params_fit,cfg_data=data_params)

    for i in data.test_dataloader():
        X,y = i[0][:,:-1], i[0][:,-1:]
        fitfunc(X,y)        

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # ,1,2,4,5,6,7" Change Me
    #print(f"Starting a run with {config}")
    main()