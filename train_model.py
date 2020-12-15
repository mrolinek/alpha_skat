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


class TrainSkatModel(pl.LightningModule):

    def __init__(self, arch_params, value_scaling_constant, scheduler_params, optimizer_params):

        super().__init__()
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params
        self.value_scaling_constant = value_scaling_constant
        self.save_hyperparameters()

        # Define PyTorch model
        self.backbone = get_arch(**arch_params, num_classes=512)
        self.policy_head = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 32))
        self.value_head = nn.Sequential(nn.Linear(512, 200), nn.ReLU(), nn.Linear(200, 3))
        self.convolutional = 'transformer' not in arch_params.name.lower()

        self.eval_accuracy = 0.0
        self.eval_loss = 0.0

        self.status_rows = 2
        self._metrics = dict(val_kl_loss=0.0, val_acc=0.0, val_mse_loss=0.0, train_loss=0.0)

    def forward(self, x):
        x = x - 0.02
        if self.convolutional:
            x = x[:, None, ...].repeat([1, 3, 1, 1])
        x = self.backbone(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    @input_to_tensors
    @output_to_numpy
    def get_policy_and_value(self, x):
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
            policy, value = self(x)
            value = value * self.value_scaling_constant

        if was_singleton:
            policy = policy[0]
            value = value[0]

        return policy, value
    
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

    def _apply_augmentation(self, inputs, masks, probs, true_state_values):
        perm = np.array(self._get_random_permutation())
        inputs[:, self.status_rows:, :] = inputs[:, self.status_rows:, perm]
        return inputs, masks[:, perm], probs[:, perm], true_state_values

    def training_step(self, batch, batch_idx):
        #inputs, masks, true_policy_probs, true_state_values = batch
        inputs, masks, true_policy_probs, true_state_values = self._apply_augmentation(*batch)

        predicted_policy_logits, predicted_values = self(inputs)
        policy_loss = F.kl_div(torch.log_softmax(predicted_policy_logits, dim=1), true_policy_probs,
                               reduction='batchmean')
        value_loss = ((predicted_values - true_state_values / self.value_scaling_constant) ** 2).mean()

        self.log('loss_v', value_loss, prog_bar=True)
        return policy_loss + value_loss

    def training_epoch_end(self, training_step_outputs):
        losses = [it['loss'] for it in training_step_outputs]
        self._metrics["train_loss"] = sum(losses) / len(losses)
        self.log('train_loss', self._metrics["train_loss"], prog_bar=False)


    def validation_step(self, batch, batch_idx):
        inputs, masks, true_policy_probs, true_state_values = batch

        predicted_policy_logits, predicted_values = self(inputs)
        policy_loss = F.kl_div(torch.log_softmax(predicted_policy_logits, dim=1), true_policy_probs, reduction='batchmean')
        value_loss = (self.value_scaling_constant*predicted_values - true_state_values).abs().mean()

        pred_ys = torch.argmax(predicted_policy_logits + 1000 * (masks-1), dim=1)
        true_ys = torch.argmax(true_policy_probs + 1000 * (masks-1), dim=1)
        acc = accuracy(pred_ys, true_ys)

        # Calling self.log will surface up scalars for you in TensorBoard
        return acc, value_loss, policy_loss

    def validation_epoch_end(self, validation_step_outputs):
        new_metrics = ["val_acc", "val_mse_loss", "val_kl_loss"]
        metrics_to_update = dict(zip(new_metrics, zip(*validation_step_outputs)))
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
        policy_probs = torch.Tensor(np.load(os.path.join(self.data_dir, "policy_probs.npy")))
        state_values = torch.Tensor(np.load(os.path.join(self.data_dir, "state_values.npy")))
        full_dataset = TensorDataset(inputs, masks, policy_probs, state_values)
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
