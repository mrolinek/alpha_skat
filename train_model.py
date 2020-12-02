import os
import random

import torch
from cluster import cluster_main
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.models import resnet18, resnet50, resnet101, resnext50_32x4d
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import metrics

import numpy as np


from pytorch_lightning.metrics import Metric


def check(x):
    assert not torch.isnan(x).any()

arch_dict = dict(resnet18=resnet18, resnet50=resnet50, resnet101=resnet101, resnext50_32x4d=resnext50_32x4d)

class TrainSkatModel(pl.LightningModule):

    def __init__(self, learning_rate, arch_name):

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Define PyTorch model
        self.model = arch_dict[arch_name](num_classes=32)

        self.eval_accuracy = 0.0
        self.eval_loss = 0.0

    def forward(self, x):
        x = x[:, None, ...].repeat([1, 3, 1, 1]) - 0.02
        x = self.model(x)
        return F.softmax(x, dim=1)

    def _logits_to_loss_and_probs(self, predicted_probs, masks, true_probs):
        corrected_probs = predicted_probs / (torch.sum(predicted_probs, dim=1, keepdim=True) + 1e-5)
        errors = torch.sum(torch.abs(corrected_probs - true_probs), dim=1, keepdim=True)
        loss = (errors / torch.sum(masks, dim=1, keepdim=True)).mean()
        return loss, corrected_probs

    def training_step(self, batch, batch_idx):
        inputs, masks, probs = batch
        predicted_probs = self(inputs) * masks
        loss, corrected_probs = self._logits_to_loss_and_probs(predicted_probs, masks, probs)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, masks, probs = batch

        predicted_probs = self(inputs) * masks
        loss, corrected_probs = self._logits_to_loss_and_probs(predicted_probs, masks, probs)
        pred_ys = torch.argmax(corrected_probs, dim=1)
        true_ys = torch.argmax(probs, dim=1)
        acc = accuracy(pred_ys, true_ys)

        # Calling self.log will surface up scalars for you in TensorBoard
        return acc, loss

    def validation_epoch_end(self, validation_step_outputs):
        accs, losses = zip(*validation_step_outputs)
        self.eval_loss = sum(losses) / len(losses)
        self.eval_accuracy = sum(accs) / len(accs)
        self.log('val_acc', self.eval_accuracy, prog_bar=True)
        self.log('val_loss', self.eval_loss, prog_bar=True)


    @property
    def metrics(self):
        return dict(eval_acc=self.eval_accuracy, eval_loss=self.eval_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class SkatDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        inputs = torch.Tensor(np.load(os.path.join(self.data_dir, "inputs.npy")))
        masks = torch.Tensor(np.load(os.path.join(self.data_dir, "masks.npy")))
        probs = torch.Tensor(np.load(os.path.join(self.data_dir, "probs.npy")))
        full_dataset = TensorDataset(inputs, masks, probs)
        length = inputs.shape[0]
        train_size = int(0.9 * length)
        self.train_set, self.val_set = random_split(full_dataset, [train_size, length - train_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=2)


@cluster_main
def main(working_dir, num_epochs, model_params, data_params):
    data_module = SkatDataModule(**data_params)
    model = TrainSkatModel(**model_params)
    trainer = pl.Trainer(gpus=1, max_epochs=num_epochs, progress_bar_refresh_rate=20, default_root_dir=working_dir)
    trainer.fit(model, data_module)
    return model.metrics


if __name__ == "__main__":
    main()
