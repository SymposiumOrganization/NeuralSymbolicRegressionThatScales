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
#from nesymres.utils import *
from nesymres.architectures.model import Model
from nesymres.architectures.data import DataModule
from nesymres.dclasses import Params, DataModuleParams
seed_everything(9)
from nesymres.utils import load_data


def main():
    train_path = "data/datasets/100K/100K_train"
    train_data = load_data(train_path)
    val_data = load_data("data/datasets/100K/100K_val_subset")
    test_data = None
    params = Params(datamodule_params_train=DataModuleParams(
                                total_variables=list(train_data.total_variables), 
                                total_coefficients=list(train_data.total_coefficients)),
                    datamodule_params_val=DataModuleParams(
                        total_variables=list(val_data.total_variables), 
                        total_coefficients=list(train_data.total_coefficients)))
    data = DataModule(
        train_data,
        val_data,
        None,
        cfg=params
    )
    model = Model(cfg=params.architecture)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss/dataloader_idx_0",
        dirpath="Exp_weights/",                 
        filename=str(train_path)+"_log_"+"-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    trainer = pl.Trainer(
        #distributed_backend="ddp",
        gpus=-1,
        max_epochs=params.max_epochs,
        val_check_interval=params.val_check_interval,
        callbacks=[checkpoint_callback],
        precision=params.precision,
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ,1,2,4,5,6,7" Change Me
    #print(f"Starting a run with {config}")
    main()