import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):

    def __init__(self, input_dim, seq_len, num_heads, dim_feedforward, num_layers, dropout, num_classes):
        super(TransformerModel, self).__init__()

        encoder_layers = TransformerEncoderLayer(input_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.readout = nn.Linear(input_dim*seq_len, num_classes)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.readout.bias.data.zero_()
        self.readout.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        batch_size = output.shape[0]
        logits = self.readout(output.reshape(batch_size, -1))
        return logits
