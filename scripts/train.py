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
from nesymres.utils import *
from nesymres.architectures.set_transformer import * 
from nesymres.dataset.data_module import * 
from nesymres.dclasses import Params
seed_everything(9)






# hyperparameter_defaults = dict(
#     ###DataModuleParams
#     n_l_enc=4,
#     max_number_of_points=500,
#     type_of_sampling_points=["Constant", "Logarithm"][1],
#     predict_c=False,    #Luca changed it on 27/01/2020
#     constant_degree_of_freedom=3,
#     support_extremes=(-10, 10),
#     distribution_support=[ "Uniform", "Gaussian"][0],
#     ###Architecture
#     sinuisodal_embeddings=False,
#     dec_pf_dim=512,
#     dec_layers=5,
#     dim_hidden=512,  
#     lr=0.0001,
#     dropout=0,
#     num_features=10,
#     ln=True,
#     N_p=0,
#     num_inds=50,
#     activation="relu",
#     bit32=True, 
#     norm=True,
#     linear=False,
#     input_normalization=False
# )

def main():
    params = Params()
    # dim_hidden = config.dim_hidden
    # n_l_enc = config.n_l_enc
    # dec_layers = config.dec_layers
    # num_heads = 8
    # dec_pf_dim = config.dec_pf_dim
    # dim_input = 4
    # output_dim = 33 
    # src_pad_index = 0
    # length_eq = 40
    # trg_pad_idx = 0
    # bit32 = config.bit32
    # mean = torch.tensor([0.5]) 
    # std = torch.tensor([0.5])  
    # input_normalization = config.input_normalization
    # data_params = DataModuleParams(
    #     config.max_number_of_points,
    #     config.type_of_sampling_points,
    #     config.support_extremes,
    #     config.constant_degree_of_freedom,
    #     config.predict_c,
    #     config.distribution_support,
    #     config.input_normalization
    # )
    # num_inds = config.num_inds
    # lr = config.lr
    # num_features = config.num_features
    # ln = config.ln
    # N_p = config.N_p
    # sizes = ["experiment/1000K"]
    # S = sizes[N_p]
    # print(S)

    data = DataModule(
        data_dir1=S,
        data_dir2="facebook/20000K_val",
        data_dir3="AIFeynman/ai_feymann",
        cfg=params
    )
    model = SetTransformer(cfg=params)
    #     n_l_enc,
    #     src_pad_idx,
    #     trg_pad_idx,
    #     dim_input,
    #     output_dim,
    #     dim_hidden,
    #     dec_layers,
    #     num_heads,
    #     dec_pf_dim,
    #     config.dropout,
    #     length_eq,
    #     lr,
    #     num_inds,
    #     ln,
    #     num_features,
    #     config.sinuisodal_embeddings,
    #     bit32,
    #     config.norm,
    #     config.activation,
    #     config.linear,
    #     mean,
    #     std,
    #     input_normalization
    # )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss/dataloader_idx_0",
        dirpath="Exp_weights/",                 
        filename=str(S)+"_log_"+"-{epoch:02d}-{val_loss:.2f}",
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