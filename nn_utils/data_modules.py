import os
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset

import pytorch_lightning as pl
import numpy as np


class CardDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)

    def _assign_split(self, split_ratio, dataset_length, dataset):
        train_size = int(split_ratio * dataset_length)
        self.train_set, self.val_set = random_split(dataset, [train_size, dataset_length - train_size])


class PolicyDataModule(CardDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        inputs = torch.Tensor(np.load(os.path.join(self.data_dir, "inputs.npy")))
        masks = torch.Tensor(np.load(os.path.join(self.data_dir, "masks.npy")))
        policy_probs = torch.Tensor(np.load(os.path.join(self.data_dir, "policy_probs.npy")))
        full_dataset = TensorDataset(inputs, masks, policy_probs)
        length = inputs.shape[0]
        self._assign_split(split_ratio=0.9, dataset_length=length, dataset=full_dataset)


class ValueDataModule(CardDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def setup(self, stage=None):
        states = torch.Tensor(np.load(os.path.join(self.data_dir, "full_states.npy")))
        values = torch.Tensor(np.load(os.path.join(self.data_dir, "full_state_values.npy")))
        full_dataset = TensorDataset(states, values)
        length = states.shape[0]
        self._assign_split(split_ratio=0.9, dataset_length=length, dataset=full_dataset)
