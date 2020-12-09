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
from nn_utils.models import TransformerModel
from torchvision.models.resnet import _resnet, Bottleneck, BasicBlock

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import metrics

import numpy as np


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

        self.status_rows = 2

    def forward(self, x):
        x = x - 0.02
        if self.convolutional:
            x = x[:, None, ...].repeat([1, 3, 1, 1])
        x = self.model(x)
        return x

    def value_loss(self, predicted_qs, masks, real_qs, squared=True):
        action_nums = torch.sum(masks, dim=1, keepdim=True)
        if squared:
            loss_v = (((torch.sum(masks*predicted_qs, dim=1) - torch.sum(real_qs, dim=1)) / action_nums) ** 2).mean()
        else:
            loss_v = (torch.max(masks*predicted_qs, dim=1)[0] - torch.max(real_qs, dim=1)[0]).abs().mean()
        return loss_v

    def policy_loss(self, predicted_qs, masks, true_qs):
        true_probs = torch.softmax(true_qs+1000*(masks-1), dim=1)*masks

        loss = F.kl_div(torch.log_softmax(predicted_qs, dim=1), true_probs, reduction='batchmean')
        return loss

    def prob_weighted_loss(self, predicted_qs, masks, true_qs):
        true_probs = torch.softmax(true_qs+1000*(masks-1), dim=1)*masks
        return (true_probs * (predicted_qs - true_qs) ** 2).sum(dim=1).mean()
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _permutation_for_i(random_integer):
        all_suit_permutations =list(itertools.permutations([0, 1, 2, 3]))
        fixed_suit_permuation = all_suit_permutations[random_integer]
        permutation = []
        for i in range(32):
            if i % 8 == 4:
                permutation.append(i)
            else:
                suit = i // 8
                new_suit = fixed_suit_permuation[suit]
                permutation.append(i % 8 + 8*new_suit)

        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        return tuple(idx)


    def _get_random_permutation(self):
        random_integer = random.randint(0, 23)
        return TrainSkatModel._permutation_for_i(random_integer)

    def _apply_augmentation(self, inputs, masks, probs):
        perm = np.array(self._get_random_permutation())
        inputs[:, self.status_rows:, :] = inputs[:, self.status_rows:, perm]
        return inputs, masks[:, perm], probs[:, perm]

    def training_step(self, batch, batch_idx):
        inputs, masks, q_values = batch
        inputs, masks, q_values = self._apply_augmentation(inputs, masks, q_values)
        predicted_qs = self(inputs)
        return self.prob_weighted_loss(predicted_qs, masks, q_values / scaling_constant)
        #loss_kl = self.policy_loss(predicted_qs, masks, q_values / scaling_constant)
        #loss_v = self.value_loss(predicted_qs, masks, q_values / scaling_constant)
        #self.log('loss_v', loss_v, prog_bar=True)
        #return loss_kl + 0.1*loss_v

    def training_epoch_end(self, training_step_outputs):
        losses = [it['loss'] for it in training_step_outputs]
        self.train_loss = sum(losses) / len(losses)
        self.log('train_loss', self.train_loss, prog_bar=False)


    def validation_step(self, batch, batch_idx):
        inputs, masks, true_qs = batch

        predicted_qs = self(inputs)
        loss = self.value_loss(predicted_qs * scaling_constant, masks, true_qs, squared=False)
        pred_ys = torch.argmax(predicted_qs + 1000 * (masks-1), dim=1)
        true_ys = torch.argmax(true_qs + 1000 * (masks-1), dim=1)
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
        q_values = torch.Tensor(np.load(os.path.join(self.data_dir, "qvalues.npy")))
        full_dataset = TensorDataset(inputs, masks, q_values)
        length = inputs.shape[0]
        train_size = int(0.9 * length)
        self.train_set, self.val_set = random_split(full_dataset, [train_size, length - train_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=2, shuffle=True)

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
