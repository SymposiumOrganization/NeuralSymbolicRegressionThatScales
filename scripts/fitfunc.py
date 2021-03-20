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
from dataclasses import dataclass
from typing import Tuple
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.dclasses import Params, DataModuleParams
seed_everything(9)
from nesymres.utils import load_data


def main():
    model_path = "Exp_weights/data/datasets/100K/100K_train_log_-epoch=02-val_loss=1.24.ckpt"
    test_data = load_data("data/datasets/100K/100K_val_subset")
    params = Params(datamodule_params_test=DataModuleParams(
                                total_variables=list(test_data.total_variables), 
                                total_coefficients=list(test_data.total_coefficients)),num_of_workers=0)

    params_fit = Params()
    data = DataModule(
        None,
        None,
        test_data,
        cfg=params
    )
    data.setup()
    model = Model.load_from_checkpoint(model_path, cfg=params.architecture)
    model.eval()
    model.cuda()
    for i in data.test_dataloader():
        X,y = i[0][:,:-1], i[0][:,-1]
        model.fitfunc(X,y)


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # ,1,2,4,5,6,7" Change Me
    #print(f"Starting a run with {config}")
    main()