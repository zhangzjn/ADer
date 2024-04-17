import time

import numpy as np
import torch
import torch.nn as nn
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.vision_transformer import VisionTransformer, _cfg, checkpoint_filter_fn
from timm.models.helpers import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from functools import partial
from timm.models.layers import trunc_normal_
from model._moco import VisionTransformerMoCo
from model import get_model
from model import MODEL

# ========== Fusion ==========
class Fusion(nn.Module):
	def __init__(self, dim, mul):
		super(Fusion, self).__init__()
		self.fc = nn.Linear(dim * mul, dim)

	def forward(self, features):
		# B, L, C
		feature_align = torch.cat(features, dim=2)
		feature_align = self.fc(feature_align)
		return feature_align

@MODEL.register_module
def fusion(pretrained=False, **kwargs):
	model = Fusion(**kwargs)
	return model

# ========== ViT Encoder ==========
# deit encoder
class DistilledVisionTransformer(VisionTransformer):
	def __init__(self, teachers, neck, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.teachers = teachers
		self.neck = neck
		self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
		num_patches = self.patch_embed.num_patches
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
		self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

		trunc_normal_(self.dist_token, std=.02)
		trunc_normal_(self.pos_embed, std=.02)
		self.head_dist.apply(self._init_weights)

	def forward(self, x):
		B = x.shape[0]
		x = self.patch_embed(x)

		cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
		dist_token = self.dist_token.expand(B, -1, -1)
		x = torch.cat((cls_tokens, dist_token, x), dim=1)

		x = x + self.pos_embed
		x = self.pos_drop(x)

		out, neck = [], []
		for idx, blk in enumerate(self.blocks):
			x = blk(x)
			fea = x[:, 2:]
			if (idx + 1) in self.neck:
				neck.append(fea)  # B, L, C
			if (idx + 1) in self.teachers:
				B, L, C = fea.shape
				H = int(np.sqrt(L))
				fea = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
				out.append(fea)

		return out, neck

# normal encoder
class ViT_Encoder(VisionTransformer):
	def __init__(self, teachers, neck, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.teachers = teachers
		self.neck = neck

	def forward(self, x):
		x = self.patch_embed(x)
		x = self._pos_embed(x)

		out_neck, out_t = [], []
		if self.grad_checkpointing and not torch.jit.is_scripting():
			x = checkpoint_seq(self.blocks, x)
		else:
			for i in range(len(self.blocks)):
				x = self.blocks[i](x)
				fea = x[:, 1:, :]
				if i+1 in self.neck:
					out_neck.append(fea)
				if i+1 in self.teachers:
					B, L, C = fea.shape
					H = int(np.sqrt(L))
					fea = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
					out_t.append(fea)

		return out_t, out_neck

# ========== ViT Decoder ==========
class ViT_Decoder(VisionTransformer):
	def __init__(self, students, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.students = students
		num_patches = self.patch_embed.num_patches
		self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
		trunc_normal_(self.pos_embed, std=.02)

	def forward(self, x):
		# taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
		# B, C, H, W = x.shape
		# x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, L, C
		x = x + self.pos_embed
		x = self.pos_drop(x)

		out = []
		for idx, blk in enumerate(self.blocks):
			x = blk(x)
			if (idx + 1) in self.students:
				fea = x
				B, L, C = fea.shape
				H = int(np.sqrt(L))
				fea = fea.view(B, H, H, C).permute(0, 3, 1, 2).contiguous()
				out.append(fea)

		return [out[int(len(out)-1-i)] for i in range(len(out))]

# ========== Encoders with Different Pre-trained Weights ==========
def _create_vision_transformer(variant, pretrained=False, **kwargs):
	if kwargs.get('features_only', None):
		raise RuntimeError('features_only not implemented for Vision Transformer models.')

	if 'flexi' in variant:
		_filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
	else:
		_filter_fn = checkpoint_filter_fn
	finish = False
	while not finish:
		try:
			model = build_model_with_cfg(ViT_Encoder, variant, pretrained, pretrained_filter_fn=_filter_fn, **kwargs)
			finish = True
		except:
			print('Try load model for ViTAD')
			time.sleep(1)
	return model

### ViT
@MODEL.register_module
def vit_small_patch16_224_1k(pretrained=True, **kwargs):
	""" ViT-Small (ViT-S/16)"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = _create_vision_transformer('vit_small_patch16_224.augreg_in1k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_base_patch16_224(pretrained=True, **kwargs):
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=12, num_heads=12)
	model = _create_vision_transformer('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_large_patch16_224(pretrained=True, **kwargs):
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=1024, depth=24, num_heads=16)
	model = _create_vision_transformer('vit_large_patch16_224.augreg_in21k_ft_in1k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_huge_patch14_224(pretrained=True, **kwargs):
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=14, embed_dim=1280, depth=32, num_heads=16)
	model = _create_vision_transformer('vit_huge_patch14_224.orig_in21k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

### DINO
@MODEL.register_module
def vit_small_patch16_224_dino(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = _create_vision_transformer('vit_small_patch16_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_small_patch8_224_dino(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch8"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=8, embed_dim=384, depth=12, num_heads=6)
	model = _create_vision_transformer('vit_small_patch8_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_base_patch16_224_dino(pretrained=True, **kwargs):
	""" ViT-Base-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=12, num_heads=12)
	model = _create_vision_transformer('vit_base_patch16_224.dino', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model


### DINOv2
@MODEL.register_module
def vit_small_patch16_224_dino2(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = _create_vision_transformer('vit_small_patch14_dinov2.lvd142m', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

@MODEL.register_module
def vit_small_patch16_224_dino2reg(pretrained=True, **kwargs):
	""" ViT-Small-DINO, patch16"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6)
	model = _create_vision_transformer('vit_small_patch14_reg4_dinov2.lvd142m', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

### DeiT
@MODEL.register_module
def deit_tiny_distilled_patch16_224(pretrained=True, **kwargs):
	model = DistilledVisionTransformer(
        img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	model.default_cfg = _cfg()
	if pretrained:
		checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
		model.load_state_dict(checkpoint["model"])
	return model

@MODEL.register_module
def deit_small_distilled_patch16_224(pretrained=True, **kwargs):
	model = DistilledVisionTransformer(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	model.default_cfg = _cfg()
	if pretrained:
		checkpoint = torch.hub.load_state_dict_from_url(
			url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
			map_location="cpu", check_hash=True
		)
		model.load_state_dict(checkpoint["model"])
	return model

### MAE
@MODEL.register_module
def mae_vit_base_patch16_dec512d8b(pretrained=True, **kwargs):
	model = ViT_Encoder(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=12, num_heads=12,
		mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	if pretrained:
		checkpoint = torch.hub.load_state_dict_from_url(
			url="https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth",
			map_location="cpu", check_hash=True
		)
		new_state_dict = {}
		for k, v in checkpoint['model'].items():
			if 'fc_norm.weight' in k:
				new_state_dict['norm.weight'] = v
			elif 'fc_norm.bias' in k:
				new_state_dict['norm.bias'] = v
			else:
				new_state_dict[k] = v
		model.load_state_dict(new_state_dict)
	return model

### MOCOv3
@MODEL.register_module
def mocov3_vit_small(pretrained=True, **kwargs):
	model = VisionTransformerMoCo(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	model.default_cfg = _cfg()
	if pretrained:
		checkpoint = torch.hub.load_state_dict_from_url(
			url="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/linear-vit-s-300ep.pth.tar",
			map_location="cpu", check_hash=True
		)
		new_state_dict = {}
		for k, v in checkpoint["state_dict"].items():
			new_state_dict[k[7:]] = v
		model.load_state_dict(new_state_dict)
	return model

@MODEL.register_module
def mocov3_vit_base(pretrained=True, **kwargs):
	model = VisionTransformerMoCo(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	model.default_cfg = _cfg()
	if pretrained:
		checkpoint = torch.hub.load_state_dict_from_url(
			url="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar",
			map_location="cpu", check_hash=True
		)
		new_state_dict = {}
		for k, v in checkpoint["state_dict"].items():
			new_state_dict[k[7:]] = v
		model.load_state_dict(new_state_dict)
	return model

### CLIP
@MODEL.register_module
def vit_base_patch16_clip_224(pretrained=True, **kwargs):
	""" ViT-B/16 CLIP image tower"""
	model_kwargs = dict(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=12, num_heads=12,
						pre_norm=True, norm_layer=nn.LayerNorm)
	model = _create_vision_transformer(
		'vit_base_patch16_clip_224.openai_ft_in12k_in1k', pretrained=pretrained, **dict(model_kwargs, **kwargs))
	return model

# ========== Decoders with Different Pre-trained Weights ==========
### ViT
@MODEL.register_module
def de_vit_small_patch16_224_1k(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=kwargs.pop('depth'), num_heads=6, **kwargs)
	return model

@MODEL.register_module
def de_vit_base_patch16_224(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=kwargs.pop('depth'), num_heads=12, **kwargs)
	return model

@MODEL.register_module
def de_vit_large_patch16_224(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=1024, depth=kwargs.pop('depth'), num_heads=16, **kwargs)
	return model

@MODEL.register_module
def de_vit_huge_patch14_224(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=14, embed_dim=1280, depth=kwargs.pop('depth'), num_heads=16, **kwargs)
	return model

### DINO
@MODEL.register_module
def de_vit_small_patch16_224_dino(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=kwargs.pop('depth'), num_heads=6, **kwargs)
	return model

@MODEL.register_module
def de_vit_base_patch16_224_dino(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=kwargs.pop('depth'), num_heads=12, **kwargs)
	return model

@MODEL.register_module
def de_vit_small_patch8_224_dino(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=8, embed_dim=384, depth=kwargs.pop('depth'), num_heads=6, **kwargs)
	return model

### DeiT
@MODEL.register_module
def de_deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
	model = ViT_Decoder(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=192, depth=kwargs.pop('depth'), num_heads=3,
		mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model

@MODEL.register_module
def de_deit_small_distilled_patch16_224(pretrained=False, **kwargs):
	model = ViT_Decoder(
		img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=kwargs.pop('depth'), num_heads=6,
		mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model

### MAE
@MODEL.register_module
def de_mae_vit_base_patch16_dec512d8b(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=kwargs.pop('depth'), num_heads=12, **kwargs)
	return model

### MOCOv3
@MODEL.register_module
def de_mocov3_vit_small(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=384, depth=kwargs.pop('depth'), num_heads=6, **kwargs)
	return model

@MODEL.register_module
def de_mocov3_vit_base(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=kwargs.pop('depth'), num_heads=12, **kwargs)
	return model

### CLIP
@MODEL.register_module
def de_vit_base_patch16_clip_224(pretrained=False, **kwargs):
	model = ViT_Decoder(img_size=kwargs.pop('img_size'), patch_size=16, embed_dim=768, depth=kwargs.pop('depth'), num_heads=12, **kwargs)
	return model

# ========== ViTAD ==========
class ViTAD(nn.Module):
	def __init__(self, model_t, model_f, model_s):
		super(ViTAD, self).__init__()
		self.net_t = get_model(model_t)
		self.net_fusion = get_model(model_f)
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
		feats_t, feats_n = self.net_t(imgs)  # list
		feats_t = [f.detach() for f in feats_t]
		feats_n = [f.detach() for f in feats_n]
		feats_s = self.net_s(self.net_fusion(feats_n))
		return feats_t, feats_s


@MODEL.register_module
def vitad(pretrained=False, **kwargs):
	model = ViTAD(**kwargs)
	return model


if __name__ == '__main__':
	from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
	from util.util import get_timepc, get_net_params
	from argparse import Namespace as _Namespace

	bs = 2
	reso = 256
	x = torch.randn(bs, 3, reso, reso).cuda()

	model_t = _Namespace()
	model_t.name = 'vit_small_patch16_224_dino'
	model_t.kwargs = dict(pretrained=True, checkpoint_path='', strict=True, img_size=reso, teachers=[3, 6, 9], neck=[12])
	model_f = _Namespace()
	model_f.name = 'fusion'
	model_f.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, dim=384, mul=1)
	model_s = _Namespace()
	model_s.name = 'de_vit_small_patch16_224_dino'
	model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, img_size=reso, students=[3, 6, 9], depth=9)
	model = _Namespace()
	model.name = 'vitad'
	model.kwargs = dict(pretrained=False, checkpoint_path='', strict=True, model_t=model_t, model_f=model_f, model_s=model_s)

	net = vitad(pretrained=False, checkpoint_path='', strict=True, model_t=model_t, model_f=model_f, model_s=model_s).cuda()
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

