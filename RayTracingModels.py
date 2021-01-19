import torch
import torch.nn as nn
import numpy as np


class DepthWiseSeparableConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 inner_kernel_size=1,
                 inner_stride=1,
                 inner_padding=0):
        """Depthwise separable 2D Convolution.

        :param in_channels: Input channels.
        :type in_channels: int
        :param out_channels: Output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape/size.
        :type kernel_size: int|tuple|list
        :param stride: Stride.
        :type stride: int|tuple|list
        :param padding: Padding.
        :type padding: int|tuple|list
        :param dilation: Dilation.
        :type dilation: int
        :param bias: Bias.
        :type bias: bool
        :param padding_mode: Padding mode.
        :type padding_mode: str
        :param inner_kernel_size: Kernel shape/size of the second convolution.
        :type inner_kernel_size: int|tuple|list
        :param inner_stride: Inner stride.
        :type inner_stride: int|tuple|list
        :param inner_padding: Inner padding.
        :type inner_padding: int|tuple|list
        """
        super(DepthWiseSeparableConvBlock, self).__init__()

        self.depth_wise_conv: nn.Module = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=(in_channels if out_channels >= in_channels else out_channels), bias=bias,
            padding_mode=padding_mode)

        self.non_linearity: nn.Module = nn.LeakyReLU()

        self.point_wise: nn.Module = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=inner_kernel_size, stride=inner_stride,
            padding=inner_padding, dilation=1,
            groups=1, bias=bias, padding_mode=padding_mode)
        if inner_kernel_size != 1:
            print("NOT USING DWS")
            print("Fix inner kernel size")
            raise ValueError
        self.layers = nn.Sequential(
            self.depth_wise_conv,
            self.non_linearity,
            self.point_wise)

    def forward(self, x):
        """Forward pass of the module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, kernelSize=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernelSize, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernelSize, padding=padding)

    def forward(self, x):
        return (self.relu(self.conv2(self.relu(self.conv1(x)))))


class Encoder(nn.Module):
    def __init__(self, kernelSize, chs, useDepthwise=True):
        super().__init__()
        self.kernelSize = kernelSize
        self.padding = list((np.array(self.kernelSize) - 1) // 2)
        self.firstLayer = Block(chs[0], chs[1])
        if useDepthwise:
            self.enc_blocks = nn.ModuleList(
                [DepthWiseSeparableConvBlock(chs[i], chs[i + 1], self.kernelSize, padding=self.padding) for i in
                 range(1, len(chs) - 1)])
        else:
            self.enc_blocks = nn.ModuleList(
                [Block(chs[i], chs[i + 1], self.kernelSize, padding=self.padding) for i in
                 range(1, len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        x = self.firstLayer(x)
        ftrs.append(x)
        x = self.pool(x)
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, kernelSize, chs, inputSize, useDepthwise=True):
        super().__init__()
        self.chs = chs
        self.kernelSize = kernelSize
        self.padding = list((np.array(self.kernelSize) - 1) // 2)
        tmp = np.array(inputSize)
        inputSizes = np.zeros((len(tmp), len(chs)), dtype=np.uint8)
        for x in reversed(range(0, len(chs))):
            val = tmp // (2 ** x)
            inputSizes[:, len(chs) - x - 1] = val
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], kernel_size=2, stride=2, padding=0,
                                                         output_padding=list(inputSizes[:, i + 1] % 2))
                                      for i in range(len(chs) - 1)])
        if useDepthwise:
            self.dec_blocks = nn.ModuleList(
                [DepthWiseSeparableConvBlock(chs[i], chs[i + 1], self.kernelSize, padding=self.padding) for i in
                 range(len(chs) - 1)])
        else:
            self.dec_blocks = nn.ModuleList(
                [Block(chs[i], chs[i + 1], self.kernelSize, padding=self.padding) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            encf = encoder_features[i]
            x = torch.cat([x, encf], dim=1)
            x = self.dec_blocks[i](x)
        return x


class UNet(nn.Module):
    def __init__(self, inputSize, kernel=[3, 7], enc_chs=(
            32, 32, 64, 128, 256),
                 dec_chs=(258, 128, 64, 32), useDepthwise=True):
        super().__init__()

        self.encoder = Encoder(kernel, enc_chs, useDepthwise=useDepthwise)

        self.decoder = Decoder(kernel, dec_chs, inputSize=inputSize, useDepthwise=useDepthwise)

        # Channel reducing 1x1 convolution
        self.head = nn.Conv2d(dec_chs[-1], enc_chs[0], 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)

        return out


class lightFieldModel(nn.Module):
    def __init__(self, projectorViews=41):
        super(lightFieldModel, self).__init__()
        # NOT SUGGESTED MODEL
        self.cnn_enc = nn.Conv2d(in_channels=3 * projectorViews, out_channels=32, kernel_size=[5, 5], padding=3)
        self.ReLU = nn.ReLU()
        self.maxP = nn.MaxPool2d(4)
        self.BN1 = nn.BatchNorm2d(32)

        # self.flatten = nn.Flatten()
        self.cnn2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=[5, 5], padding=3)
        self.BN2 = nn.BatchNorm2d(32)

        # self.linear = nn.Linear(80 // 4 * 80 // 4 * 32, 80 // 4 * 80 // 4 * 32)
        self.upsamp = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4,
                                         padding=[4 + 1, 4 + 1])
        self.cnn_dec = nn.Conv2d(in_channels=32, out_channels=3 * projectorViews, kernel_size=[5, 5], padding=3)
        # self.BN3 = nn.BatchNorm2d(3*projectorViews)

        self.model = nn.Sequential(self.cnn_enc, self.ReLU, self.maxP, self.BN1, self.cnn2, self.ReLU, self.BN2,
                                   self.upsamp, self.cnn_dec, self.ReLU)

    def forward(self, x):
        return x + self.model(x)


class fullyConnectedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(600 * 800, 150)
        self.hidden_layer2 = nn.Linear(150, 600 * 800)
        self.ReLU = nn.ReLU()
        self.flat = nn.Flatten(start_dim=2)

    def forward(self, x):
        x2 = self.flat(x)
        x2 = self.ReLU(self.hidden_layer2(self.ReLU(self.hidden_layer(x2))))

        return torch.reshape(x2, (1, 123, 600, 800))
