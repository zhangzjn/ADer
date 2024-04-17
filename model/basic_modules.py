import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from functools import partial

from einops import rearrange, reduce
from timm.layers.weight_init import trunc_normal_
from timm.layers.activations import *
from timm.models.layers import DropPath
from timm.models._efficientnet_builder import _parse_ksize


# ========== For Common ==========
class LayerNorm2d(nn.Module):
	
	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
		super().__init__()
		self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
	
	def forward(self, x):
		x = rearrange(x, 'b c h w -> b h w c').contiguous()
		x = self.norm(x)
		x = rearrange(x, 'b h w c -> b c h w').contiguous()
		return x


def get_norm(norm_layer='in_1d'):
	eps = 1e-6
	norm_dict = {
		'none': nn.Identity,
		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
		'gn': partial(nn.GroupNorm, eps=eps),
		'ln_1d': partial(nn.LayerNorm, eps=eps),
		'ln_2d': partial(LayerNorm2d, eps=eps),
	}
	return norm_dict[norm_layer]


def get_act(act_layer='relu'):
	act_dict = {
		'none': nn.Identity,
		'sigmoid': Sigmoid,
		'swish': Swish,
		'mish': Mish,
		'hsigmoid': HardSigmoid,
		'hswish': HardSwish,
		'hmish': HardMish,
		'tanh': Tanh,
		'relu': nn.ReLU,
		'relu6': nn.ReLU6,
		'prelu': PReLU,
		'gelu': GELU,
		'silu': nn.SiLU
	}
	return act_dict[act_layer]


class LayerScale(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=True):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))
	
	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2D(nn.Module):
	def __init__(self, dim, init_values=1e-5, inplace=True):
		super().__init__()
		self.inplace = inplace
		self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))
	
	def forward(self, x):
		return x.mul_(self.gamma) if self.inplace else x * self.gamma
	
	
class ConvNormAct(nn.Module):
	
	def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
				 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
		super(ConvNormAct, self).__init__()
		self.has_skip = skip and dim_in == dim_out
		padding = math.ceil((kernel_size - stride) / 2)
		self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
		self.norm = get_norm(norm_layer)(dim_out)
		self.act = get_act(act_layer)(inplace=inplace)
		self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()
	
	def forward(self, x):
		shortcut = x
		x = self.conv(x)
		x = self.norm(x)
		x = self.act(x)
		if self.has_skip:
			x = self.drop_path(x) + shortcut
		return x


if __name__ == '__main__':
	x = torch.rand(2, 3, 224).cuda()
