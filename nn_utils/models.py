import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from torchvision.models.resnet import _resnet, BasicBlock


class TransformerModel(nn.Module):

    def __init__(self, input_dim, seq_len, num_heads, dim_feedforward, num_layers, dropout, num_classes, resnet_params):
        super(TransformerModel, self).__init__()

        encoder_layers = TransformerEncoderLayer(input_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.readout = _resnet('resnet', BasicBlock, pretrained=False, progress=None,
                               num_classes=num_classes, **resnet_params)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        output = output[:, None, ...].repeat([1, 3, 1, 1])
        logits = self.readout(output)
        return logits
