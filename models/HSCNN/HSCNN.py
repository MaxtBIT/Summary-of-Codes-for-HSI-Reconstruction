# coding: utf-8


import torch
from torchsummary import summary
from models.HSCNN.layers import ReLU, Leaky, Swish, Mish, FReLU


class HSCNN(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, feature=64, layer_num=9, **kwargs):
        super(HSCNN, self).__init__()
        activation = kwargs.get('activation', 'relu')
        mode = kwargs.get('mode', 'add')
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish, 'frelu': FReLU}
        self.start_conv = torch.nn.Conv2d(input_ch, output_ch, 3, 1, 1)
        self.start_activation = activations[activation]()
        self.patch_extraction = torch.nn.Conv2d(output_ch, feature, 3, 1, 1)
        self.feature_map = torch.nn.ModuleList([torch.nn.Conv2d(feature, feature, 3, 1, 1) for _ in range(layer_num - 1)])
        self.activations = torch.nn.ModuleList([activations[activation]() for _ in range(layer_num - 1)])
        self.residual_conv = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)

    def forward(self, x):

        x = self.start_conv(x)
        # x_in = self.residual_shortcut(x)
        x_in = x
        x = self.start_activation(self.patch_extraction(x))
        for feature_map, activation in zip(self.feature_map, self.activations):
            x = activation(feature_map(x))
        output = self.residual_conv(x) + x_in
        return output


if __name__ == '__main__':

    model = HSCNN(1, 31, activation='sin2')
    summary(model, (1, 64, 64))
