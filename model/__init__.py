import glob
import importlib

import torch
import torch.nn as nn
from timm.models._registry import is_model_in_modules
from timm.models._helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models._hub import load_model_config_from_hf
from timm.models._factory import parse_model_name
from util.registry import Registry
MODEL = Registry('Model')

def get_model(cfg_model):
	model_name = cfg_model.name
	kwargs = {k: v for k, v in cfg_model.kwargs.items()}
	model_fn = MODEL.get_module(model_name)
	pretrained = kwargs.pop('pretrained')
	checkpoint_path = kwargs.pop('checkpoint_path')
	strict = kwargs.pop('strict')

	if model_name.startswith('timm_'):
		if 'hf' in kwargs:
			model_name_hf = kwargs.pop('hf')
		else:
			model_name_hf = None
		if model_name_hf:
			pretrained_cfg, model_name_hf = load_model_config_from_hf(model_name_hf)
			pretrained_cfg['url'] = ''
		else:
			pretrained_cfg = None
		with set_layer_config(scriptable=None, exportable=None, no_jit=None):
			model = model_fn(pretrained=pretrained, pretrained_cfg=pretrained_cfg, **kwargs)
		if not pretrained and pretrained_cfg is None and checkpoint_path is not None:
			load_checkpoint(model, checkpoint_path, strict=strict)
	else:
		model = model_fn(pretrained=pretrained, **kwargs)
		if checkpoint_path:
			ckpt = torch.load(checkpoint_path, map_location='cpu')
			if 'net' in ckpt.keys():
				state_dict = ckpt['net']
			else:
				state_dict = ckpt
			if not strict and False:
				no_ft_keywords = model.no_ft_keywords()
				for no_ft_keyword in no_ft_keywords:
					del state_dict[no_ft_keyword]
				ft_head_keywords, num_classes = model.ft_head_keywords()
				for ft_head_keyword in ft_head_keywords:
					if state_dict[ft_head_keyword].shape[0] <= num_classes:
						del state_dict[ft_head_keyword]
					elif state_dict[ft_head_keyword].shape[0] == num_classes:
						continue
					else:
						# state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes, :] if 'weight' in ft_head_keyword else state_dict[ft_head_keyword][:num_classes]
						state_dict[ft_head_keyword] = state_dict[ft_head_keyword][:num_classes]
			# check classifier, if not match, then re-init classifier to zero
			# if 'head.bias' in state_dict:
			# 	head_bias_pretrained = state_dict['head.bias']
			# 	Nc1 = head_bias_pretrained.shape[0]
			# 	Nc2 = model.head.bias.shape[0]
			# 	if (Nc1 != Nc2):
			# 		if Nc1 == 21841 and Nc2 == 1000:
			# 			map22kto1k_path = f'data/map22kto1k.txt'
			# 			with open(map22kto1k_path) as f:
			# 				map22kto1k = f.readlines()
			# 			map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
			# 			state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
			# 			state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
			# 		else:
			# 			torch.nn.init.constant_(model.head.bias, 0.)
			# 			torch.nn.init.constant_(model.head.weight, 0.)
			# 			del state_dict['head.weight']
			# 			del state_dict['head.bias']
			# load ckpt
			if isinstance(model, nn.Module):
				model.load_state_dict(state_dict, strict=strict)
			else:
				for sub_model_name, sub_state_dict in state_dict.items():
					sub_model = getattr(model, sub_model_name, None)
					sub_model.load_state_dict(sub_state_dict, strict=strict) if sub_model else None
	return model

files = glob.glob('model/[!_]*.py')
for file in files:
	model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))
