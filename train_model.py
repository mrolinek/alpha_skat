import os
import random

import torch
from cluster import cluster_main
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.models import resnet18, resnet50, resnet101, resnext50_32x4d, mobilenet_v2
from efficientnet_pytorch import EfficientNet
from nn_utils.models import TransformerModel
from torchvision.models.resnet import _resnet, Bottleneck, BasicBlock

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import metrics

import numpy as np


from pytorch_lightning.metrics import Metric


def check(x):
    assert not torch.isnan(x).any()

arch_dict = dict(resnet18=resnet18, resnet50=resnet50, resnet101=resnet101, resnext50_32x4d=resnext50_32x4d,
                 mobilenet_v2=mobilenet_v2, TransformerModel=TransformerModel)

def get_efficient_net(name, num_classes, **arch_params):
    return EfficientNet.from_name(name, num_classes=num_classes, **arch_params)



def get_arch(name, num_classes, **arch_params):
    if 'efficientnet' in name.lower():
        return get_efficient_net(name, num_classes=num_classes, **arch_params)
    elif name == 'resnet9':
        return _resnet('resnet', BasicBlock, pretrained=False, progress=None,
                               num_classes=num_classes, layers=[1, 1, 1, 1])
    else:
        return arch_dict[name](num_classes=num_classes, **arch_params)



class TrainSkatModel(pl.LightningModule):

    def __init__(self, learning_rate, arch_params):

        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        # Define PyTorch model
        self.model = get_arch(**arch_params, num_classes=32)
        self.convolutional = 'transformer' not in arch_params.name.lower()

        self.eval_accuracy = 0.0
        self.eval_loss = 0.0

    def forward(self, x):
        x = x - 0.02
        if self.convolutional:
            x = x[:, None, ...].repeat([1, 3, 1, 1])
        x = self.model(x)
        return F.softmax(x, dim=1)

    def _logits_to_loss_and_probs(self, predicted_probs, masks, true_probs):
        masked_probs = predicted_probs * masks
        corrected_probs = masked_probs / (torch.sum(masked_probs, dim=1, keepdim=True) + 1e-5)
        errors = torch.sum(torch.abs(corrected_probs - true_probs), dim=1, keepdim=True)
        loss = (errors / torch.sum(masks, dim=1, keepdim=True)).mean()
        return loss, corrected_probs

    def _logits_to_loss_and_probs2(self, predicted_probs, masks, true_probs):
        loss = torch.abs(predicted_probs - true_probs).mean()
        return loss, predicted_probs

    def _logits_to_loss_and_probs3(self, predicted_probs, masks, true_probs):
        loss = F.kl_div(torch.log(predicted_probs+1e-5), true_probs, reduction='batchmean')
        return loss, predicted_probs

    def training_step(self, batch, batch_idx):
        inputs, masks, probs = batch
        predicted_probs = self(inputs)
        loss, corrected_probs = self._logits_to_loss_and_probs3(predicted_probs, masks, probs)
        return loss

    def training_epoch_end(self, training_step_outputs):
        losses = [it['loss'] for it in training_step_outputs]
        self.train_loss = sum(losses) / len(losses)
        self.log('train_loss', self.train_loss, prog_bar=False)


    def validation_step(self, batch, batch_idx):
        inputs, masks, probs = batch

        predicted_probs = self(inputs)
        loss, corrected_probs = self._logits_to_loss_and_probs3(predicted_probs, masks, probs)
        pred_ys = torch.argmax(corrected_probs * masks, dim=1)
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
        return dict(eval_acc=self.eval_accuracy, eval_loss=self.eval_loss, train_loss=self.train_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        return [optimizer], [scheduler]


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
    print(model.metrics)
    return model.metrics


if __name__ == "__main__":
    main()
