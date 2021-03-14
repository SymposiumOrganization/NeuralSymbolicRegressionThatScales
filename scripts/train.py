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
from nesymres.dclasses import Params
seed_everything(9)

def main():
    params = Params()
    train_path = "data/datasets/20K/20K_train"
    data = DataModule(
        env_path="config.json",
        train_path=train_path,
        val_path="data/datasets/20K/20K_val_subset",
        test_path=None,
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