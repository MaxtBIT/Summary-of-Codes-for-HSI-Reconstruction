# coding: UTF-8


import torch


class UNetConv2d(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, kernel_size=3, stride=1, padding=None, **kwargs):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.activation = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(input_ch, output_ch, kernel_size, stride, padding=padding)

    def forward(self, x):
        return self.activation(self.conv(x))


class UNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, layer_num=6, deeper=False, **kwargs):

        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        encode_feature_num = [2 ** (5 + i) for i in range(layer_num)]
        decode_feature_num = encode_feature_num[::-1]
        self.deeper = deeper
        self.skip_encode_feature = [2, 7, 12, 17, 21]
        self.input_conv = UNetConv2d(input_ch, encode_feature_num[0], 3, 1)
        self.encoder = torch.nn.ModuleList([
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[0], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[1], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[2], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                UNetConv2d(encode_feature_num[3], encode_feature_num[3], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[3], encode_feature_num[4], 3, 1),
                UNetConv2d(encode_feature_num[4], encode_feature_num[4], 3, 1),
                UNetConv2d(encode_feature_num[4], encode_feature_num[4], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[4], encode_feature_num[5], 3, 1),
                ])
        self.bottleneck1 = UNetConv2d(encode_feature_num[-1], encode_feature_num[-1], 3, 1)
        self.bottleneck2 = UNetConv2d(encode_feature_num[-1], decode_feature_num[0], 3, 1)
        self.skip_decode_feature = [1, 4, 8, 12, 16]
        self.decoder = torch.nn.ModuleList([
                UNetConv2d(decode_feature_num[0], decode_feature_num[0], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[0], decode_feature_num[1], 2, 2),
                UNetConv2d(decode_feature_num[1] + encode_feature_num[-2],
                          decode_feature_num[1], 3, 1, 1),
                UNetConv2d(decode_feature_num[1], decode_feature_num[1], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[1], decode_feature_num[2], 2, 2),
                UNetConv2d(decode_feature_num[2] + encode_feature_num[-3],
                           decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[2], decode_feature_num[3], 2, 2),
                UNetConv2d(decode_feature_num[3] + encode_feature_num[-4],
                           decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[3], decode_feature_num[4], 2, 2),
                UNetConv2d(decode_feature_num[4] + encode_feature_num[-5],
                           decode_feature_num[4], 3, 1),
                UNetConv2d(decode_feature_num[4], decode_feature_num[4], 3, 1),
                UNetConv2d(decode_feature_num[4], decode_feature_num[4], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[4], decode_feature_num[5], 2, 2),
                UNetConv2d(decode_feature_num[5] + encode_feature_num[-6],
                           decode_feature_num[5], 3, 1),
                UNetConv2d(decode_feature_num[5], decode_feature_num[5], 3, 1),
                UNetConv2d(decode_feature_num[5], decode_feature_num[5], 3, 1),
                ])
        self.output_conv = UNetConv2d(decode_feature_num[-1], output_ch, 3, 1)


    def forward(self, x):
        x = self.input_conv(x)
        encode_feature = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_encode_feature:
                encode_feature.append(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        j = 1
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i in self.skip_decode_feature:
                x = torch.cat([x, encode_feature[-j]], dim=1)
                j += 1
        x = self.output_conv(x)

        return x


class RefineUNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, layer_num=4, deeper=False, **kwargs):

        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        encode_feature_num = [2 ** (5 + i) for i in range(layer_num)]
        decode_feature_num = encode_feature_num[::-1]
        self.deeper = deeper
        self.skip_encode_feature = [2, 7, 12, 17, 21]
        self.input_conv = UNetConv2d(input_ch, encode_feature_num[0], 3, 1)
        self.encoder = torch.nn.ModuleList([
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                UNetConv2d(encode_feature_num[0], encode_feature_num[0], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[0], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                UNetConv2d(encode_feature_num[1], encode_feature_num[1], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[1], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                UNetConv2d(encode_feature_num[2], encode_feature_num[2], 3, 1),
                torch.nn.MaxPool2d((2, 2)),
                UNetConv2d(encode_feature_num[2], encode_feature_num[3], 3, 1),
                ])
        self.bottleneck1 = UNetConv2d(encode_feature_num[-1], encode_feature_num[-1], 3, 1)
        self.bottleneck2 = UNetConv2d(encode_feature_num[-1], decode_feature_num[0], 3, 1)
        self.skip_decode_feature = [1, 4, 8, 12, 16]
        self.decoder = torch.nn.ModuleList([
                UNetConv2d(decode_feature_num[0], decode_feature_num[0], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[0], decode_feature_num[1], 2, 2),
                UNetConv2d(decode_feature_num[1] + encode_feature_num[-2],
                          decode_feature_num[1], 3, 1, 1),
                UNetConv2d(decode_feature_num[1], decode_feature_num[1], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[1], decode_feature_num[2], 2, 2),
                UNetConv2d(decode_feature_num[2] + encode_feature_num[-3],
                           decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                UNetConv2d(decode_feature_num[2], decode_feature_num[2], 3, 1),
                torch.nn.ConvTranspose2d(decode_feature_num[2], decode_feature_num[3], 2, 2),
                UNetConv2d(decode_feature_num[3] + encode_feature_num[-4],
                           decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1),
                UNetConv2d(decode_feature_num[3], decode_feature_num[3], 3, 1)
                ])
        self.output_conv = UNetConv2d(decode_feature_num[-1], output_ch, 3, 1)


    def forward(self, x):
        x = self.input_conv(x)
        encode_feature = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_encode_feature:
                encode_feature.append(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        j = 1
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i in self.skip_decode_feature:
                x = torch.cat([x, encode_feature[-j]], dim=1)
                j += 1
        x = self.output_conv(x)

        return x


class Discriminator(torch.nn.Module):

    def __init__(self, input_ch, input_h, input_w, *args, layer_num=4, **kwargs):
        super().__init__()
        activation = kwargs.get('activation', 'relu').lower()
        feature_num = [input_ch] + [2 ** (6 + i) for i in range(layer_num)]
        output_h, output_w = input_h // (2 ** (layer_num)), input_w // (2 ** (layer_num))
        self.conv_layers = torch.nn.ModuleList([UNetConv2d(feature_num[i], feature_num[i + 1], kernel_size=3, stride=2)
                                                for i in range(layer_num)])
        self.fc = torch.nn.Linear(output_h * output_w * feature_num[-1], 1)
        self.output_activation = torch.nn.Sigmoid()

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.output_activation(self.fc(x))
        return x


class LambdaNet(torch.nn.Module):

    def __init__(self, input_ch, output_ch, *args, layer_num=6, deeper=False, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self.ReconstStage = UNet(input_ch, output_ch)
        self.RefineStage = RefineUNet(output_ch, output_ch)

    def forward(self, x):
        return self.RefineStage(self.ReconstStage(x))

    def load_Reconst(self, path, key='model_state_dict'):
        ckpt = torch.load(path, map_location=self.device)
        self.ReconstStage.load_state_dict(ckpt[key])
        return self

    def load_Refine(self, path, key='model_state_dict'):
        ckpt = torch.load(path, map_location=self.device)
        self.RefineStage.load_state_dict(ckpt[key])
        return self
