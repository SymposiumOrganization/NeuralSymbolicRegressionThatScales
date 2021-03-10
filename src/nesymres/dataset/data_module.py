import os
import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import *
from dataset import wrapper_dataset
from data_utils import *


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir1,
        data_dir2,
        data_dir3,
        data_params,
        batch_size=20,
        num_of_workers=0,
    ):
        super().__init__()
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.data_dir3 = data_dir3
        self.batch = batch_size
        self.data_params = data_params
        self.num_of_workers = num_of_workers

    def setup(self, stage=None):
        """called one ecah GPU separately - stage defines if we are at fit or test step"""
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == "fit" or stage is None:
            data_train = load_data(self.data_dir1)
            env,_ = create_env(self.data_dir1)
            data_val = load_data(self.data_dir2)
            self.training_dataset = wrapper_dataset(
                data_train,
                env,
                self.data_params,
            )
            self.validation_dataset = wrapper_dataset(
                data_val,
                env,
                self.data_params,
            )
            data_test = load_data(self.data_dir3)
            self.test_dataset = wrapper_dataset(
                data_test, env, self.data_params
            )

    def train_dataloader(self):
        """returns training dataloader"""
        trainloader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
            collate_fn=custom_collate_fn,
            num_workers=self.num_of_workers,
            pin_memory=True
        )
        return trainloader

    def val_dataloader(self):
        """returns validation dataloader"""
        validloader = torch.utils.data.DataLoader(
            self.validation_dataset,
            batch_size=self.batch,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_of_workers,
            pin_memory=True
        )
        testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch,
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=self.num_of_workers,
            pin_memory=True
        )
        return [validloader, testloader]