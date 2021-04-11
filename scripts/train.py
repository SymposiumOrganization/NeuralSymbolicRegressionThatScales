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
from nesymres.dclasses import Params, DataModuleParams, ArchitectureParams, ConstantsOptions
from nesymres.utils import load_metadata_hdf5
import wandb
from dataclass_dict_convert import dataclass_dict_convert 
from pytorch_lightning.loggers import WandbLogger
import hydra
from pathlib import Path

@hydra.main(config_name="config")
def main(cfg):
    seed_everything(9)
    train_path = Path(hydra.utils.to_absolute_path(cfg.train_path))
    val_path = Path(hydra.utils.to_absolute_path(cfg.val_path))
    # test_data = None
    # wandb = None

    # train_constant =  ConstantsOptions(max_constants=cfg.dataset_train.num_constants,
    #                  min_additive_constant_support=cfg.dataset_train.additive.min,
    #                  min_additive_constant_support=cfg.dataset_train.additive.min,
    #                 min_additive_constant_support=cfg.dataset_train.additive.min,
    #                 min_additive_constant_support=cfg.dataset_train.additive.min)

    #     train_constant =  ConstantsOptions(max_constants=cfg.dataset_train.num_constants,
    #                  min_additive_constant_support=cfg.dataset_train.additive.min,
    #                  min_additive_constant_support=cfg.dataset_train.additive.min,
    #                 min_additive_constant_support=cfg.dataset_train.additive.min,
    #                 min_additive_constant_support=cfg.dataset_train.additive.min)


    # params = Params(datamodule_params_train=DataModuleParams(
    #                             #constant_options= train_constant
    #                             total_variables=list(train_data.total_variables), 
    #                             total_coefficients=list(train_data.total_coefficients),
    #                             # max_number_of_points=cfg.dataset_train.max_number_of_points,
    #                             # type_of_sampling_points=cfg.dataset_train.type_of_sampling_points,
    #                             # predict_c=cfg.dataset_train.predict_c,
    #                             ),

    #                 datamodule_params_val=DataModuleParams(
    #                     # constant_options= train_constant
    #                     total_variables=list(val_data.total_variables), 
    #                     total_coefficients=list(val_data.total_coefficients)),
    #                     # max_number_of_points=cfg.dataset_val.max_number_of_points,
    #                     # type_of_sampling_points=cfg.dataset_val.type_of_sampling_points,
    #                     # predict_c=cfg.dataset_val.predict_c),
    #                     num_of_workers=cfg.num_of_workers)

    architecture_params = ArchitectureParams()
    data = DataModule(
        # hydra.utils.to_absolute_path(cfg.train_path),
        # hydra.utils.to_absolute_path(cfg.val_path),
        # None,
        cfg
    )
    model = Model(cfg=architecture_params)
    if wandb:
        wandb.init(config=architecture_params.to_dict(), project="ICML")
        config = wandb.config
        wandb_logger = WandbLogger()
    else:
        wandb_logger = None
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", #/dataloader_idx_0",
        dirpath="Exp_weights/",                 
        filename=train_path.stem+"_log_"+"-{epoch:02d}-{val_loss:.2f}",
        mode="min",
    )

    trainer = pl.Trainer(
        distributed_backend="ddp",
        gpus=-1,
        max_epochs=cfg.epochs,
        val_check_interval=cfg.val_check_interval,
        precision=cfg.precision,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # ,1,2,4,5,6,7" Change Me
    main()