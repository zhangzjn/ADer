import torch
import torch.nn as nn
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL


# ========== Decoder ==========
def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
	return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)


class DeBasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
		dilation = 1, norm_layer = None):
		super(DeBasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		if stride == 2:
			self.conv1 = deconv2x2(inplanes, planes, stride)
		else:
			self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.upsample is not None:
			identity = self.upsample(x)

		out += identity
		out = self.relu(out)

		return out


class DeBottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride = 1, upsample = None, groups = 1, base_width = 64,
		dilation = 1, norm_layer = None):
		super(DeBottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		if stride == 2:
			self.conv2 = deconv2x2(width, width, stride, groups, dilation)
		else:
			self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.upsample = upsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.upsample is not None:
			identity = self.upsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):

	def __init__(self, block, layers, num_classes = 1000,
				 zero_init_residual = False, groups = 1, width_per_group = 64, replace_stride_with_dilation = None,
				 norm_layer = None ):
		super(ResNet, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 512 * block.expansion
		self.dilation = 1
		if replace_stride_with_dilation is None:
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group
		self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, DeBottleneck):
					nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
				elif isinstance(m, DeBasicBlock):
					nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		upsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			upsample = nn.Sequential(deconv2x2(self.inplanes, planes * block.expansion, stride),
									 norm_layer(planes * block.expansion),)
		layers = []
		layers.append(block(self.inplanes, planes, stride, upsample, self.groups, self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		feature_a = self.layer1(x)  # 512*8*8->256*16*16
		feature_b = self.layer2(feature_a)  # 256*16*16->128*32*32
		feature_c = self.layer3(feature_b)  # 128*32*32->64*64*64
		return [feature_c, feature_b, feature_a]

	def forward(self, x):
		return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
	model = ResNet(block, layers, **kwargs)
	# if pretrained:
		# state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
		# model.load_state_dict(state_dict)
	return model

@MODEL.register_module
def de_resnet18(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet18', DeBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet34(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet34', DeBasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet50(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet50', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet101(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet101', DeBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnet152(pretrained = False, progress = True, **kwargs):
	return _resnet('resnet152', DeBottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnext50_32x4d(pretrained = False, progress = True, **kwargs):
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 4
	return _resnet('resnext50_32x4d', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_resnext101_32x8d(pretrained = False, progress = True, **kwargs):
	kwargs['groups'] = 32
	kwargs['width_per_group'] = 8
	return _resnet('resnext101_32x8d', DeBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_wide_resnet50_2(pretrained = False, progress = True, **kwargs):
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet50_2', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_wide_resnet50_3(pretrained = False, progress = True, **kwargs):
	kwargs['width_per_group'] = 64 * 2
	# return _resnet('wide_resnet50_3', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
	return _resnet('wide_resnet50_3', DeBottleneck, [6, 4, 3, 3], pretrained, progress, **kwargs)

@MODEL.register_module
def de_wide_resnet101_2(pretrained = False, progress = True, **kwargs):
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet101_2', DeBottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


# ========== MFF & OCE ==========
class MFF_OCE(nn.Module):
	def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
		super(MFF_OCE, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer
		self.base_width = width_per_group
		self.inplanes = 256 * block.expansion
		self.dilation = 1
		self.bn_layer = self._make_layer(block, 512, layers, stride=2)

		self.conv1 = conv3x3(64 * block.expansion, 128 * block.expansion, 2)
		self.bn1 = norm_layer(128 * block.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
		self.bn2 = norm_layer(256 * block.expansion)
		self.conv3 = conv3x3(128 * block.expansion, 256 * block.expansion, 2)
		self.bn3 = norm_layer(256 * block.expansion)
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(conv1x1(self.inplanes*3, planes * block.expansion, stride),
									   norm_layer(planes * block.expansion), )
		layers = []
		layers.append(block(self.inplanes*3, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		l1 = self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x[0]))))))
		l2 = self.relu(self.bn3(self.conv3(x[1])))
		feature = torch.cat([l1,l2,x[2]],1)
		output = self.bn_layer(feature)

		return output.contiguous()

	def forward(self, x):
		return self._forward_impl(x)


class RD(nn.Module):
	def __init__(self, model_t, model_s):
		super(RD, self).__init__()
		self.net_t = get_model(model_t)
		self.mff_oce = MFF_OCE(Bottleneck, 3)
		self.net_s = get_model(model_s)

		self.frozen_layers = ['net_t']

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
		feats_t = self.net_t(imgs)
		feats_t = [f.detach() for f in feats_t]
		feats_s = self.net_s(self.mff_oce(feats_t))
		return feats_t, feats_s

@MODEL.register_module
def rd(pretrained=False, **kwargs):
	model = RD(**kwargs)
	return model


if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
	from util.util import get_timepc, get_net_params
	from argparse import Namespace as _Namespace

	bs = 2
	reso = 256
	x = torch.randn(bs, 3, reso, reso).cuda()

	model_t = _Namespace()
	model_t.name = 'timm_wide_resnet50_2'
	model_t.kwargs = dict(pretrained=False, checkpoint_path='model/pretrain/wide_resnet50_racm-8234f177.pth', strict=False, features_only=True, out_indices=[1, 2, 3])
	model_s = _Namespace()
	model_s.name = 'de_wide_resnet50_2'
	model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=True)

	net = RD(model_t, model_s).cuda()
	net.eval()
	y = net(x)

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
