# coding: utf-8


import torch
from .layers import swish, mish, HSI_prior_block


class DeepSSPrior(torch.nn.Module):

    def __init__(self, input_ch, output_ch, feature=64, block_num=9, activation='relu', output_norm=None, **kwargs):
        super(DeepSSPrior, self).__init__()
        self.activation = activation
        self.output_norm = output_norm
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)
        self.start_shortcut = torch.nn.Identity()
        hsi_block = [HSI_prior_block(output_ch, output_ch, feature=feature) for _ in range(block_num)]
        self.hsi_block = torch.nn.Sequential(*hsi_block)
        self.residual_block = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)
        self.output_conv = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.start_conv(x)
        x_in = x
        for hsi_block in self.hsi_block:
            x_hsi = hsi_block(x)
            x_res = self.residual_block(x)
            x = x_res + x_hsi + x_in
        return self._output_norm_fn(self.output_conv(x))

    def _activation_fn(self, x):
        if self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)

    def _output_norm_fn(self, x):
        if self.output_norm == 'sigmoid':
            return torch.sigmoid(x)
        elif self.output_norm == 'tanh':
            return torch.tanh(x)
        else:
            return x
