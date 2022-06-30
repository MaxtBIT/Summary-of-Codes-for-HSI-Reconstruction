# coding: utf-8


import torch
import numpy  as np


# ########################## Activation Function ##########################


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


class FReLU(torch.nn.Module):

    def __init__(self, input_ch, output_ch, kernel_size, stride=1, **kwargs):
        super(FReLU, self).__init__()
        self.depth = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2, groups=input_ch)

    def forward(self, x):
        depth = self.depth(x)
        return torch.max(x, depth)


class ReLU(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(ReLU, self).__init__()
        pass

    def forward(self, x):
        return torch.relu(x)


class Leaky(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Leaky, self).__init__()
        self.alpha = kwargs.get('alpha', .02)

    def forward(self, x):
        return torch.where(x < 0., x * self.alpha, x)


class Swish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Swish, self).__init__()
        pass

    def forward(self, x):
        return swish(x)


class Mish(torch.nn.Module):

    def __init__(self, *args, **kwargs):
        super(Mish, self).__init__()
        pass

    def forward(self, x):
        return mish(x)


# ########################## Loss Function ##########################


class SAMLoss(torch.nn.Module):

    def forward(self, x, y):
        x_sqrt = torch.norm(x, dim=1)
        y_sqrt = torch.norm(y, dim=1)
        xy = torch.sum(x * y, dim=1)
        metrics = xy / (x_sqrt * y_sqrt + 1e-6)
        angle = torch.acos(metrics)
        return torch.mean(angle)


class MSE_SAMLoss(torch.nn.Module):

    def __init__(self, alpha=.5, beta=.5, mse_ratio=1., sam_ratio=.01):
        super(MSE_SAMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = torch.nn.MSELoss()
        self.sam_loss = SAMLoss()
        self.mse_ratio = mse_ratio
        self.sam_ratio = sam_ratio

    def forward(self, x, y):
        return self.alpha * self.mse_ratio * self.mse_loss(x, y) + self.beta * self.sam_ratio * self.sam_loss(x, y)


# ########################## Loss Function ##########################


class Base_Module(torch.nn.Module):

    def __init__(self):
        super(Base_Module, self).__init__()

    def _activation_fn(self, x):
        if self.activation == 'swish':
            return swish(x)
        elif self.activation == 'mish':
            return mish(x)
        elif self.activation == 'leaky' or self.activation == 'leaky_relu':
            return torch.nn.functional.leaky_relu(x)
        # elif self.activation == 'frelu':
        #     return FReLU(x)
        elif self.activation == 'relu':
            return torch.relu(x)
        else:
            return x


class DW_PT_Conv(Base_Module):

    def __init__(self, input_ch, output_ch, kernel_size, activation='relu'):
        super(DW_PT_Conv, self).__init__()
        self.activation = activation
        self.depth = torch.nn.Conv2d(input_ch, input_ch, kernel_size, 1, 1, groups=input_ch)
        self.point = torch.nn.Conv2d(input_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x = self.depth(x)
        x = self._activation_fn(x)
        x = self.point(x)
        x = self._activation_fn(x)
        return x


class HSI_prior_block(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(HSI_prior_block, self).__init__()
        self.activation = activation
        self.spatial_1 = torch.nn.Conv2d(input_ch, feature, 3, 1, 1)
        self.spatial_2 = torch.nn.Conv2d(feature, output_ch, 3, 1, 1)
        self.spectral = torch.nn.Conv2d(output_ch, input_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


class My_HSI_network(Base_Module):

    def __init__(self, input_ch, output_ch, feature=64, activation='relu'):
        super(My_HSI_network, self).__init__()
        self.activation = activation
        self.spatial_1 = DW_PT_Conv(input_ch, feature, 3, activation=None)
        self.spatial_2 = DW_PT_Conv(feature, output_ch, 3, activation=None)
        self.spectral = torch.nn.Conv2d(output_ch, output_ch, 1, 1, 0)

    def forward(self, x):
        x_in = x
        h = self.spatial_1(x)
        h = self._activation_fn(h)
        h = self.spatial_2(h)
        x = h + x_in
        x = self.spectral(x)
        return x


# ########################## Ghost Mix Layers ##########################


def split_layer(output_ch, chuncks):
    split = [np.int(np.ceil(output_ch / chuncks)) for _ in range(chuncks)]
    split[chuncks - 1] += output_ch - sum(split)
    return split


class Ghost_layer(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, kernel_size=3, stride=1, dw_kernel=3, dw_stride=1, ratio=2, **kwargs):
        super(Ghost_layer, self).__init__()
        self.ratio = ratio
        self.mode = kwargs.get('mode', None).lower()
        chuncks = kwargs.get('chuncks', ratio - 1)
        self.output_ch = output_ch
        primary_ch = int(np.ceil(output_ch / ratio))
        cheap_ch = primary_ch * (self.ratio - 1)
        self.activation = kwargs.get('activation', 'relu').lower()
        self.primary_conv = torch.nn.Conv2d(input_ch, primary_ch, kernel_size, stride, padding=kernel_size // 2)
        if self.mode == 'mix1':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_Conv(cheap_ch, cheap_ch, chuncks=chuncks)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix])
        elif self.mode == 'mix2':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_Conv(cheap_ch, cheap_ch, chuncks=chuncks)
            pw = torch.nn.Conv2d(cheap_ch, cheap_ch, 1, 1, 0)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix, pw])
        elif self.mode == 'mix3':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_SS_Layer(cheap_ch, cheap_ch, chuncks=chuncks, groups=ratio - 1)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix])
        elif self.mode == 'mix4':
            cheap_conv_before = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                                dw_stride, dw_kernel // 2, groups=primary_ch)
            mix = Mix_SS_Layer(cheap_ch, cheap_ch, chuncks=chuncks, groups=ratio - 1)
            pw = torch.nn.Conv2d(cheap_ch, cheap_ch, 1, 1, 0)
            self.cheap_conv = torch.nn.Sequential(*[cheap_conv_before, mix, pw])
        else:
            self.cheap_conv = torch.nn.Conv2d(primary_ch, cheap_ch, dw_kernel,
                                              dw_stride, padding=dw_kernel // 2, groups=primary_ch)

    def forward(self, x):
        primary_x = self.primary_conv(x)
        cheap_x = self.cheap_conv(primary_x)
        output = torch.cat([primary_x, cheap_x], dim=1)
        return output[:, :self.output_ch, :, :]


class GroupConv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, kernel_size, *args, stride=1, **kwargs):
        super(GroupConv, self).__init__()
        activation = kwargs.get('activation', 'relu').lower()
        self.activation = ReLU()
        self.chuncks = chuncks
        self.split_input_ch = split_layer(input_ch, chuncks)
        self.split_output_ch = split_layer(output_ch, chuncks)

        if chuncks == 1:
            self.group_conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride=stride, padding=kernel_size // 2)
        else:
            self.group_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_input_ch[idx],
                                                                     self.split_output_ch[idx],
                                                                     kernel_size, stride=stride,
                                                                     padding=kernel_size // 2) for idx in range(chuncks)])

    def forward(self, x):
        if self.chuncks == 1:
            return self.activation(self.group_conv(x))
        else:
            split = torch.chunk(x, self.chuncks, dim=1)
            return self.activation(torch.cat([group_layer(split_x) for group_layer, split_x in zip(self.group_layers, split)], dim=1))


class Mix_Conv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, stride=1, **kwargs):
        super(Mix_Conv, self).__init__()

        self.chuncks = chuncks
        self.output_split = split_layer(output_ch, chuncks)
        self.input_split = split_layer(input_ch, chuncks)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.input_split[idx],
                                                                self.output_split[idx],
                                                                kernel_size=2 * idx + 3,
                                                                stride=stride,
                                                                padding=(2 * idx + 3) // 2,
                                                                groups=self.input_split[idx]) for idx in range(chuncks)])

    def forward(self, x):
        split = torch.chunk(x, self.chuncks, dim=1)
        output = torch.cat([conv_layer(split_x) for conv_layer, split_x in zip(self.conv_layers, split)], dim=1)
        return output


class Group_SE(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, kernel_size, **kwargs):
        super(Group_SE, self).__init__()
        ratio = kwargs.get('ratio', 2)
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        if ratio == 1:
            feature_num = input_ch
            self.squeeze = torch.nn.Sequential()
        else:
            feature_num = max(1, output_ch // ratio)
            self.squeeze = GroupConv(input_ch, feature_num, chuncks, kernel_size, 1, 0)
        self.extention = GroupConv(feature_num, output_ch, chuncks, kernel_size, 1, 0)
        self.activation = activations[activation]()

    def forward(self, x):
        gap = torch.mean(x, [2, 3], keepdim=True)
        squeeze = self.activation(self.squeeze(gap))
        extention = self.extention(squeeze)
        return torch.sigmoid(extention) * x


class Mix_Conv(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, stride=1, **kwargs):
        super(Mix_Conv, self).__init__()

        self.chuncks = chuncks
        self.split_layer = split_layer(output_ch, chuncks)
        self.conv_layers = torch.nn.ModuleList([torch.nn.Conv2d(self.split_layer[idx],
                                                                self.split_layer[idx],
                                                                kernel_size=2 * idx + 3,
                                                                stride=stride,
                                                                padding=(2 * idx + 3) // 2,
                                                                groups=self.split_layer[idx]) for idx in range(chuncks)])

    def forward(self, x):
        split = torch.chunk(x, self.chuncks, dim=1)
        output = torch.cat([conv_layer(split_x) for conv_layer, split_x in zip(self.conv_layers, split)], dim=1)
        return output


class Mix_SS_Layer(torch.nn.Module):

    def __init__(self, input_ch, output_ch, chuncks, *args, feature_num=64, group_num=1, **kwargs):
        super(Mix_SS_Layer, self).__init__()
        activations = {'relu': ReLU, 'leaky': Leaky, 'swish': Swish, 'mish': Mish}
        activation = kwargs.get('activation', 'relu').lower()
        se_flag = kwargs.get('se_flag', False)
        self.spatial_conv = GroupConv(input_ch, feature_num, group_num, kernel_size=3, stride=1)
        self.mix_conv = Mix_Conv(feature_num, feature_num, chuncks)
        self.se_block = Group_SE(feature_num, feature_num, chuncks, kernel_size=1) if se_flag is True else torch.nn.Sequential()
        self.spectral_conv = GroupConv(feature_num, output_ch, group_num, kernel_size=1, stride=1)
        self.shortcut = torch.nn.Sequential()
        self.spatial_activation = activations[activation]()
        self.mix_activation = activations[activation]()


    def forward(self, x):
        h = self.spatial_activation(self.spatial_conv(x))
        h = self.mix_activation(self.mix_conv(h))
        h = self.se_block(h)
        h = self.spectral_conv(h)
        return h + self.shortcut(x)
