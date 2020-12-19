import itertools
import os
import random
from functools import lru_cache

import torch
from cluster import cluster_main
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision.models import resnet18, resnet50, resnet101, resnext50_32x4d, mobilenet_v2
from efficientnet_pytorch import EfficientNet

from nn_utils.game_constants import constants
from nn_utils.models import TransformerModel
from torchvision.models.resnet import _resnet, Bottleneck, BasicBlock

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import metrics

import numpy as np

from utils import input_to_tensors, output_to_numpy

scaling_constant = 50.0


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


def get_optimizer(parameters, name, **optimizer_params):
    opt_dict = dict(Adam=torch.optim.Adam,
                    AdamW=torch.optim.AdamW,
                    SGD=torch.optim.SGD,
                    RMSProp=torch.optim.RMSprop)
    return opt_dict[name](parameters, **optimizer_params)


class CardGameModel(pl.LightningModule):

    def __init__(self, scheduler_params, optimizer_params, game):

        super().__init__()
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.save_hyperparameters()
        self.backbone = None

        self.value_scaling_constant = constants[game]["value_scaling_constant"]
        self.status_rows = constants[game]["status_rows"]

    @staticmethod
    @lru_cache(maxsize=32)
    def _permutation_for_i(random_integer):
        all_suit_permutations = list(itertools.permutations([0, 1, 2, 3]))
        fixed_suit_permuation = all_suit_permutations[random_integer]
        permutation = []
        for i in range(32):
            if i % 8 == 4:
                permutation.append(i)
            else:
                suit = i // 8
                new_suit = fixed_suit_permuation[suit]
                permutation.append(i % 8 + 8 * new_suit)

        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        return tuple(idx)

    def _get_random_permutation(self):
        random_integer = random.randint(0, 23)
        return CardGameModel._permutation_for_i(random_integer)

    def training_epoch_end(self, training_step_outputs):
        losses = [it['loss'] for it in training_step_outputs]
        self._metrics["train_loss"] = sum(losses) / len(losses)
        self.log('train_loss', self._metrics["train_loss"], prog_bar=False)

    def validation_epoch_end(self, validation_step_outputs):
        metrics_to_update = dict(zip(self.validation_metrics, zip(*validation_step_outputs)))
        for key, value in metrics_to_update.items():
            metrics_to_update[key] = sum(value) / len(value)
            self.log(key, metrics_to_update[key], prog_bar=True)
        self._metrics.update(metrics_to_update)

    @property
    def metrics(self):
        return self._metrics

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(), **self.optimizer_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]


class PolicyModel(CardGameModel):
    def __init__(self, *, arch_params, **kwargs):
        super().__init__(**kwargs)
        self.backbone = get_arch(**arch_params, num_classes=32)
        self.convolutional = 'transformer' not in arch_params.name.lower()
        self.validation_metrics = ["val_acc", "val_kl_loss"]
        self.train_metrics = ["train_loss"]
        self._metrics = {m: 0.0 for m in self.train_metrics + self.validation_metrics}

    def forward(self, x):
        x = x - 0.025
        if self.convolutional:
            x = x[:, None, ...].repeat([1, 3, 1, 1])
        policy_logits = self.backbone(x)
        return policy_logits

    @input_to_tensors
    @output_to_numpy
    def get_policy(self, x):
        was_singleton = False
        if x.ndim == 2:
            was_singleton = True
            x = x[None, ...]

        if torch.cuda.is_available() and x.shape[0] > 20:
            self.cuda()
            x = x.cuda()
        else:
            self.to('cpu')

        self.eval()
        with torch.no_grad():
            policy_logits = self(x)

        if was_singleton:
            policy_logits = policy_logits[0]

        return policy_logits

    def _apply_augmentation(self, inputs, masks, probs):
        perm = np.array(self._get_random_permutation())
        inputs[:, self.status_rows:, :] = inputs[:, self.status_rows:, perm]
        return inputs, masks[:, perm], probs[:, perm]

    def training_step(self, batch, batch_idx):
        # inputs, masks, true_policy_probs, true_state_values = batch
        inputs, masks, true_policy_probs = self._apply_augmentation(*batch)

        predicted_policy_logits = self(inputs)
        policy_loss = F.kl_div(torch.log_softmax(predicted_policy_logits, dim=1), true_policy_probs,
                               reduction='batchmean')

        return policy_loss

    def validation_step(self, batch, batch_idx):
        inputs, masks, true_policy_probs = batch

        predicted_policy_logits = self(inputs)
        policy_loss = F.kl_div(torch.log_softmax(predicted_policy_logits, dim=1), true_policy_probs,
                               reduction='batchmean')

        pred_ys = torch.argmax(predicted_policy_logits + 1000 * (masks - 1), dim=1)
        true_ys = torch.argmax(true_policy_probs + 1000 * (masks - 1), dim=1)
        acc = accuracy(pred_ys, true_ys)

        # Calling self.log will surface up scalars for you in TensorBoard
        return acc, policy_loss


class ValueModel(CardGameModel):
    def __init__(self, *, arch_params, loss_function, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = loss_function
        self.backbone = get_arch(**arch_params, num_classes=3)
        self.convolutional = 'transformer' not in arch_params.name.lower()
        self.validation_metrics = ["val_loss", "val_l1_scaled"]
        self.train_metrics = ["train_loss"]
        self._metrics = {m: 0.0 for m in self.train_metrics + self.validation_metrics}

    def forward(self, x):
        x = x - 0.035
        if self.convolutional:
            x = x[:, None, ...].repeat([1, 3, 1, 1])
        values = self.backbone(x)
        return values

    @input_to_tensors
    @output_to_numpy
    def get_value(self, x):
        was_singleton = False
        if x.ndim == 2:
            was_singleton = True
            x = x[None, ...]

        if torch.cuda.is_available() and x.shape[0] > 20:
            self.cuda()
            x = x.cuda()
        else:
            self.to('cpu')

        self.eval()
        with torch.no_grad():
            values = self(x)

        if was_singleton:
            values = values[0]

        return values

    def _apply_augmentation(self, states, values):
        perm = np.array(self._get_random_permutation())
        states[:, self.status_rows:, :] = states[:, self.status_rows:, perm]
        return states, values

    def training_step(self, batch, batch_idx):
        states, values = self._apply_augmentation(*batch)

        predicted_values = self(states)
        loss = self.loss_fn(predicted_values, values)
        return loss

    def loss_fn(self, predicted_values, true_values):
        if self.loss_function == "L2":
            return ((predicted_values - true_values / self.value_scaling_constant) ** 2).mean()
        elif self.loss_function == 'SmoothL1':
            return F.smooth_l1_loss(predicted_values, true_values / self.value_scaling_constant)

    def validation_step(self, batch, batch_idx):
        states, values = batch
        predicted_values = self(states)
        loss = self.loss_fn(predicted_values, values)
        value_l1_loss = (self.value_scaling_constant * predicted_values - values).abs().mean()

        # Calling self.log will surface up scalars for you in TensorBoard
        return loss, value_l1_loss