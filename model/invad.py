"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from argparse import Namespace

from model.stylegan2_op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from torchvision.models.resnet import BasicBlock, Bottleneck
from model.basic_modules import get_norm, get_act, ConvNormAct
from model import get_model
from model import MODEL


def _repeat_tuple(t, n):
    r"""Repeat each element of `t` for `n` times.
    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in t for _ in range(n))


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, batch):
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        lr_mul=1,
        bias=True,
        bias_init=0,
        conv_transpose2d=False,
        activation=False,
    ):
        super().__init__()

        self.out_channel = out_channel
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size).div_(lr_mul)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2) * lr_mul

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))

            self.lr_mul = lr_mul
        else:
            self.lr_mul = None

        self.conv_transpose2d = conv_transpose2d

        if activation:
            self.activation = ScaledLeakyReLU(0.2)
            # self.activation = FusedLeakyReLU(out_channel)
        else:
            self.activation = False

    def forward(self, input):
        if self.lr_mul != None:
            bias = self.bias * self.lr_mul
        else:
            bias = None

        if self.conv_transpose2d:
            # out = F.conv_transpose2d(
            #     input,
            #     self.weight.transpose(0, 1) * self.scale,
            #     bias=bias,
            #     stride=self.stride,
            #     # padding=self.padding,
            #     padding=0,
            # )

            # group version for fast training
            batch, in_channel, height, width = input.shape
            input_temp = input.view(1, batch * in_channel, height, width)
            weight = self.weight.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(
                input_temp,
                weight * self.scale,
                bias=bias,
                padding=self.padding,
                stride=2,
                groups=batch,
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            out = F.conv2d(
                input,
                self.weight * self.scale,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
            )

        if self.activation:
            out = self.activation(out)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        lr_mul=1.,
    ):
        assert not (upsample and downsample)
        layers = []

        if upsample:
            stride = 2
            self.padding = 0
            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                    conv_transpose2d=True,
                    lr_mul=lr_mul,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor))

        else:

            if downsample:
                factor = 2
                p = (len(blur_kernel) - factor) + (kernel_size - 1)
                pad0 = (p + 1) // 2
                pad1 = p // 2

                layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

                stride = 2
                self.padding = 0

            else:
                stride = 1
                self.padding = kernel_size // 2

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], return_features=False
    ):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )
        self.return_features = return_features

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)

        skip = self.skip(input)
        out = (out2 + skip) / math.sqrt(2)

        if self.return_features:
            return out, out1, out2
        else:
            return out


class Discriminator(nn.Module):
    def __init__(self, size, in_cha, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(in_cha, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        out = self.final_conv(out)
        out = out.view(batch, -1)

        out = self.final_linear(out)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        blur_kernel,
        normalize_mode,
        upsample=False,
        activate=True,
        modulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            normalize_mode=normalize_mode,
            modulate=modulate,
        )

        if activate:
            self.activate = FusedLeakyReLU(out_channel)
        else:
            self.activate = None

    def forward(self, input, style):
        out = self.conv(input, style)

        if self.activate is not None:
            out = self.activate(out)
        return out


class ModulatedConv2d(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        normalize_mode,
        blur_kernel,
        upsample=False,
        downsample=False,
        modulate=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.modulate = modulate

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.normalize_mode = normalize_mode
        if normalize_mode == "InstanceNorm2d":
            self.norm = nn.InstanceNorm2d(in_channel, affine=False)
        elif normalize_mode == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(in_channel, affine=False)

        if modulate:
            self.gamma = EqualConv2d(
                style_dim,
                in_channel,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                bias_init=1,
            )

            self.beta = EqualConv2d(
                style_dim,
                in_channel,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=True,
                bias_init=0,
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, stylecode=None):
        batch, in_channel, height, width = input.shape
        # repeat_size = input.shape[3] // stylecode.shape[3]

        weight = self.scale * self.weight
        weight = weight.repeat(batch, 1, 1, 1, 1)

        if self.normalize_mode in ["InstanceNorm2d", "BatchNorm2d"]:
            input = self.norm(input)
        elif self.normalize_mode == "LayerNorm":
            input = nn.LayerNorm(input.shape[1:], elementwise_affine=False)(input)
        elif self.normalize_mode == "GroupNorm":
            input = nn.GroupNorm(2 ** 3, input.shape[1:], affine=False)(input)
        elif self.normalize_mode == None:
            pass
        else:
            print("not implemented normalization")

        if self.modulate:
            gamma = self.gamma(stylecode)
            beta = self.beta(stylecode)
            input = input * gamma + beta

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class StyledResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, blur_kernel, normalize_mode, upsample=True, act_layer='none'):
        super().__init__()
        # self.conv1 = StyledConv(in_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, upsample=upsample, normalize_mode=None, )
        self.conv1 = StyledConv(in_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, upsample=upsample, normalize_mode=normalize_mode, modulate=not upsample)
        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, upsample=False, normalize_mode=normalize_mode, modulate=True)
        self.skip = ConvLayer(in_channel, out_channel, 1, upsample=upsample, activate=False, bias=False)
        self.acti = get_act(act_layer=act_layer)()

    def forward(self, input, stylecodes):
        out = self.conv1(input, stylecodes[0])
        out = self.conv2(out, stylecodes[1])
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        out = self.acti(out)
        return out


class NormalResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, lr_mul=1., act_layer='none'):
        super().__init__()
        self.conv1 = ConvLayer(in_channel, out_channel, kernel_size, upsample=upsample, downsample=downsample, blur_kernel=blur_kernel, bias=bias, activate=activate, lr_mul=lr_mul)
        self.conv2 = ConvLayer(out_channel, out_channel, kernel_size, upsample=False, downsample=False, blur_kernel=blur_kernel, bias=bias, activate=activate, lr_mul=lr_mul)
        self.skip = ConvLayer(in_channel, out_channel, 1, upsample=False, downsample=False, blur_kernel=blur_kernel, bias=bias, activate=activate, lr_mul=lr_mul)
        self.acti = get_act(act_layer=act_layer)()

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        out = self.acti(out)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        in_chas=[256, 512, 1024],
        style_chas=[128, 256, 256],
        latent_spatial_size=16,
        latent_channel_size=64,
        blur_kernel=[1, 3, 3, 1],
        normalize_mode='LayerNorm',
        lr_mul=0.01,
        small_generator=False,
        layers=[2, 2, 2]
    ):
        super().__init__()

        map_dim_pre = in_chas[-1]
        self.input = ConstantInput(map_dim_pre, size=latent_spatial_size)
        self.conv1 = nn.Sequential(*[
            ConvLayer(map_dim_pre, map_dim_pre, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul),
            # ConvLayer(map_dim_pre, map_dim_pre, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul),
            # StyledConv(map_dim_pre, map_dim_pre, 3, latent_channel_size, blur_kernel=blur_kernel, normalize_mode=normalize_mode, modulate=False)
        ])

        self.convs_latent = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.depth = len(in_chas)
        for i in range(self.depth):
            latent_cha = latent_channel_size if small_generator else style_chas[self.depth - i - 1]
            self.convs_latent.append(nn.Sequential(*[
                ConvLayer(style_chas[self.depth - i - 1], style_chas[self.depth - i - 1], 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul),
                ConvLayer(style_chas[self.depth - i - 1], latent_cha, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul),
            ]))

            map_dim_cur = in_chas[self.depth - 1 - i]
            convs_depth = nn.ModuleList()
            convs_depth.append(StyledResBlock(map_dim_pre, map_dim_cur, latent_cha, blur_kernel, normalize_mode=normalize_mode, upsample=False if i == 0 else True))
            for j in range(layers[self.depth - 1 - i] - 1):
                convs_depth.append(StyledResBlock(map_dim_cur, map_dim_cur, latent_cha, blur_kernel, normalize_mode=normalize_mode, upsample=False, act_layer='none'))
            self.convs.append(convs_depth)
            map_dim_pre = map_dim_cur

    def forward(self, style_codes):
        batch = style_codes[-1].shape[0]
        out = self.input(batch)
        out = self.conv1(out)
        outs = []
        for i in range(self.depth):
            style_code = self.convs_latent[i](style_codes[i])
            for j in range(len(self.convs[i])):
                if i > 0 and j == 0:
                    out = self.convs[i][j](out, [None, style_code])
                else:
                    out = self.convs[i][j](out, [style_code, style_code])
            outs.append(out)
        return outs


class Fuser(nn.Module):
    def __init__(self, in_chas=[256, 512, 1024], style_chas=[128, 256, 256], in_strides=[4, 2, 1], down_conv=True, out_stride=1, bottle_num=1, conv_num=1, conv_type='conv', lr_mul=0.01):
        super(Fuser, self).__init__()
        self.stage_num = len(in_chas)
        assert len(in_chas) == len(in_strides)
        if down_conv:
            self.downsamples = nn.ModuleList([ConvNormAct(in_cha, in_cha, kernel_size=in_stride, stride=in_stride, bias=True, norm_layer='bn_2d', act_layer='none') for in_cha, in_stride in zip(in_chas, in_strides)])
        else:
            self.downsamples = nn.ModuleList([nn.AvgPool2d(kernel_size=in_stride, stride=in_stride) for in_cha, in_stride in zip(in_chas, in_strides)])
        self.conv_cat = ConvNormAct(sum(in_chas), style_chas[-1], kernel_size=1, stride=1, norm_layer='bn_2d', act_layer='relu')
        self.conv_bottle = nn.Identity() if bottle_num < 1 else nn.Sequential(*[Bottleneck(style_chas[-1], style_chas[-1] // Bottleneck.expansion) for i in range(bottle_num)])

        self.convs = nn.ModuleList()
        for i in range(self.stage_num):
            if i == 0:
                dim_pre, dim_cur, upsample = style_chas[-1], style_chas[-1], False
            else:
                dim_pre, dim_cur, upsample = dim_cur, style_chas[len(in_chas) - i - 1], True
            if conv_type == 'conv':
                convs = [ConvLayer(dim_pre, dim_cur, 3, upsample=upsample, bias=True, activate=True, lr_mul=lr_mul)]
                for j in range(conv_num):
                    convs.append(ConvLayer(dim_cur, dim_cur, 3, upsample=False, bias=True, activate=True, lr_mul=lr_mul))
            elif conv_type == 'normresblcok':
                convs = [ConvLayer(dim_pre, dim_cur, 3, upsample=upsample, bias=True, activate=True, lr_mul=lr_mul)]
                for j in range(conv_num):
                    convs.append(NormalResBlock(dim_cur, dim_cur, kernel_size=3, upsample=False, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True, lr_mul=1., act_layer='none'))
            else:
                convs = []
            self.convs.append(nn.Sequential(*convs))

    def forward(self, feats):
        feat_list = []
        for i in range(self.stage_num):
            feat_list.append(self.downsamples[i](feats[i]))
        feat_cat = torch.cat(feat_list, dim=1)
        feat_cat = self.conv_cat(feat_cat)
        feat_bottle = self.conv_bottle(feat_cat)

        x, xs = feat_bottle, []
        for i in range(self.stage_num):
            x = self.convs[i](x)
            xs.append(x)
        return xs


class MultiScaleFuser(nn.Module):
    def __init__(self, in_chas=[256, 512, 1024], style_chas=[128, 256, 256], in_strides=[4, 2, 1], bottle_num=1, cross_reso=True):
        super(MultiScaleFuser, self).__init__()
        self.cross_reso = cross_reso
        assert len(in_chas) == len(in_strides)
        self.convs = nn.ModuleList()
        self.bottles = nn.ModuleList()
        for i in range(len(in_chas)):
            self.convs.append(ConvNormAct(in_chas[i], style_chas[i], kernel_size=1, stride=1, norm_layer='bn_2d', act_layer='relu'))
            self.bottles.append(nn.Identity() if bottle_num < 1 else nn.Sequential(*[Bottleneck(style_chas[i], style_chas[i] // Bottleneck.expansion) for i in range(bottle_num)]))
            if cross_reso:
                for j in range(len(in_chas)):
                    if j == i:
                        cross_module = nn.Identity()
                    elif j < i:
                        cross_module = ConvNormAct(style_chas[j], style_chas[i], kernel_size=2**(i - j), stride=2**(i - j), norm_layer='bn_2d', act_layer='relu')
                    else:
                        cross_module = nn.Sequential(*[
                            nn.Upsample(scale_factor=2**(j - i), mode='bicubic'),
                            ConvNormAct(style_chas[j], style_chas[i], kernel_size=1, stride=1, norm_layer='bn_2d', act_layer='relu')
                        ])
                    self.__setattr__(f'cross_{i}_{j}', cross_module)

    def forward(self, feats):
        xs = []
        for i in range(len(feats)):
            xs.append(self.bottles[i](self.convs[i](feats[i])))
        if self.cross_reso:
            xs_cross = []
            for i in range(len(xs)):
                x_cross = 0
                for j in range(len(xs)):
                    x_cross += self.__getattr__(f'cross_{i}_{j}')(xs[j])
                xs_cross.append(x_cross)
            xs = xs_cross
        xs = xs[::-1]
        return xs


def get_fuser(model_fuser):
    fuser_type = model_fuser.pop('type')
    if fuser_type == 'Fuser':
        fuser = Fuser
    elif fuser_type == 'MultiScaleFuser':
        fuser = MultiScaleFuser
    else:
        raise NotImplementedError
    return fuser(**model_fuser)


def get_disor(model_disor):
    sizes = model_disor.pop('sizes')
    in_chas = model_disor.pop('in_chas')
    net_disor = nn.ModuleList()
    for size, in_cha in zip(sizes, in_chas):
        net_disor.append(
            Discriminator(size, in_cha, channel_multiplier=2, blur_kernel=[1, 3, 3, 1])
        )
    return net_disor


class InvAD(nn.Module):
    def __init__(self, model_encoder, model_fuser, model_decoder, model_disor=None):
        super(InvAD, self).__init__()
        self.net_encoder = get_model(model_encoder)
        self.net_fuser = get_fuser(model_fuser)
        self.net_decoder = Decoder(**model_decoder)
        if model_disor:
            self.net_disor = get_disor(model_disor)

        self.frozen_layers = ['net_encoder']

    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats = self.net_encoder(imgs)
        feats_fusion = self.net_fuser(feats)
        feats_pred = self.net_decoder(feats_fusion)
        feats_pred = feats_pred[::-1]
        feats_pred = feats_pred[:len(feats)]
        return feats, feats_pred

    def forward_d(self, imgs, detach=False):
        preds = []
        for i in range(len(imgs)):
            preds.append(self.net_disor[i](imgs[i].detach() if detach else imgs[i]))
        return preds


@MODEL.register_module
def invad(pretrained=False, **kwargs):
    model = InvAD(**kwargs)
    return model


if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    from util.util import get_timepc, get_net_params
    from argparse import Namespace as _Namespace

    bs = 2
    reso = 256
    x = torch.randn(bs, 3, reso, reso).cuda()

    # ==> timm_wide_resnet50_2
    in_chas = [256, 512, 1024]  # [64, 256, 512, 1024, 2048]
    out_cha = 64
    style_chas = [min(in_cha, out_cha) for in_cha in in_chas]
    in_strides = [2 ** (len(in_chas) - i - 1) for i in range(len(in_chas))]
    latent_channel_size = 16
    model_encoder = Namespace()
    model_encoder.name = 'timm_wide_resnet50_2'
    model_encoder.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth',
                                strict=False, features_only=True, out_indices=[1, 2, 3])
    model_fuser = dict(
        type='Fuser', in_chas=in_chas, style_chas=style_chas, in_strides=[4, 2, 1], down_conv=True, bottle_num=1, conv_num=1, conv_type='conv', lr_mul=0.01)
        # type='MultiScaleFuser', in_chas=in_chas, style_chas=style_chas, in_strides=in_strides, bottle_num=1, cross_reso=True)

    latent_spatial_size = reso // (2 ** 4)
    model_decoder = dict(in_chas=in_chas, style_chas=style_chas,
                         latent_spatial_size=latent_spatial_size, latent_channel_size=latent_channel_size,
                         blur_kernel=[1, 3, 3, 1], normalize_mode='LayerNorm',
                         lr_mul=0.01, small_generator=True, layers=[2] * len(in_chas))
    sizes = [reso // (2 ** (2 + i)) for i in range(len(in_chas))]
    # model_disor = dict(sizes=sizes, in_chas=in_chas)
    model_disor = None
    net = invad(model_encoder=model_encoder, model_fuser=model_fuser, model_decoder=model_decoder, model_disor=model_disor).cuda()
    net.eval()
    y = net(x)
    # preds = net.forward_d(y[0])

    Flops = FlopCountAnalysis(net, x)
    print(flop_count_table(Flops, max_depth=5))
    flops = Flops.total() / bs / 1e9
    params = parameter_count(net)[''] / 1e6
    with torch.no_grad():
        pre_cnt, cnt = 5, 10
        for _ in range(pre_cnt):
            y = net(x)
        t_s = get_timepc()
        for _ in range(cnt):
            y = net(x)
        t_e = get_timepc()
    print('[GFLOPs: {:>6.3f}G]\t[Params: {:>6.3f}M]\t[Speed: {:>7.3f}]\n'.format(flops, params, bs * cnt / (t_e - t_s)))
# print(flop_count_table(FlopCountAnalysis(fn, x), max_depth=3))
