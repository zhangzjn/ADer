import os
import random
import shutil
import copy
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import math
import time
from collections import Iterable
from timm.utils.agc import adaptive_clip_grad
from util.util import log_msg
from fvcore.nn import FlopCountAnalysis, flop_count_table
from timm.utils import NativeScaler, ApexScaler
from contextlib import suppress, contextmanager


def init_training(cfg):
	# ---------- cudnn ----------
	if not torch.cuda.is_available():
		print('==> GPU error')
		exit(0)
	torch.cuda.empty_cache()
	if cfg.trainer.cuda_deterministic:  # slower, more reproducible
		cudnn.deterministic = True
		cudnn.benchmark = False
	else:  # faster, less reproducible
		cudnn.deterministic = False
		cudnn.benchmark = True
	# ---------- dist ----------
	cfg.dist = True
	cfg.world_size, cfg.rank, cfg.local_rank = 1, 0, 0
	cfg.ngpus_per_node = torch.cuda.device_count()
	if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
		cfg.world_size = int(os.environ['WORLD_SIZE'])
		cfg.rank = int(os.environ["RANK"])
		cfg.local_rank = int(os.environ['LOCAL_RANK'])
		cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
	elif 'SLURM_PROCID' in os.environ and os.environ['SLURM_JOB_GPUS'] != '0':
		cfg.rank = int(os.environ['SLURM_PROCID'])
		cfg.local_rank = cfg.rank % cfg.ngpus_per_node
		cfg.nnodes = cfg.world_size // cfg.ngpus_per_node
	else:
		cfg.dist = False
		cfg.nnodes = 1
	if cfg.dist:
		torch.cuda.set_device(cfg.local_rank)
		cfg.master = cfg.rank == cfg.logger_rank
		cfg.dist_backend = 'nccl'
		torch.distributed.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url, timeout=datetime.timedelta(seconds=7200 * 10))
		torch.distributed.barrier()
	else:
		cfg.master = True
	# ---------- seed ----------
	seed = cfg.seed + cfg.local_rank
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	# ---------- dataset ----------
	if cfg.trainer.data.batch_size:
		cfg.trainer.data.batch_size_per_gpu = cfg.trainer.data.batch_size // cfg.world_size
		assert cfg.trainer.data.batch_size_per_gpu * cfg.world_size == cfg.trainer.data.batch_size
	else:
		cfg.trainer.data.batch_size = cfg.trainer.data.batch_size_per_gpu * cfg.world_size
	if cfg.trainer.data.batch_size_test:
		cfg.trainer.data.batch_size_per_gpu_test = cfg.trainer.data.batch_size_test // cfg.world_size
		assert cfg.trainer.data.batch_size_per_gpu_test * cfg.world_size == cfg.trainer.data.batch_size_test
	else:
		cfg.trainer.data.batch_size_test = cfg.trainer.data.batch_size_per_gpu_test * cfg.world_size
	cfg.trainer.data.num_workers = cfg.trainer.data.num_workers_per_gpu * cfg.world_size


def init_modules(modules, w_init='xavier_normal'):
	if w_init == "normal":
		_init = torch.nn.init.normal_
	elif w_init == "xavier_normal":
		_init = torch.nn.init.xavier_normal_
	elif w_init == "xavier_uniform":
		_init = torch.nn.init.xavier_uniform_
	elif w_init == "kaiming_normal":
		_init = torch.nn.init.kaiming_normal_
	elif w_init == "kaiming_uniform":
		_init = torch.nn.init.kaiming_uniform_
	elif w_init == "orthogonal":
		_init = torch.nn.init.orthogonal_
	else:
		raise NotImplementedError
	if isinstance(modules, Iterable):
		for m in modules:
			if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
				_init(m.weight)
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)
			if isinstance(m, (nn.LSTM, nn.GRU)):
				for name, param in m.named_parameters():
					if 'bias' in name:
						nn.init.zeros_(param)
					elif 'weight' in name:
						_init(param)


def trans_state_dict(state_dict, dist=True):
	state_dict_modify = dict()
	if dist:
		for k, v in state_dict.items():
			k = k if k.startswith('module') else 'module.'+k
			state_dict_modify[k] = v
	else:
		for k, v in state_dict.items():
			k = k[7:] if k.startswith('module') else k
			state_dict_modify[k] = v
	return state_dict_modify


def dispatch_clip_grad(parameters, value, mode='norm', norm_type=2.0):
	if mode == 'norm':
		torch.nn.utils.clip_grad_norm_(parameters, value, norm_type=norm_type)
	elif mode == 'value':
		torch.nn.utils.clip_grad_value_(parameters, value)
	elif mode == 'agc':
		adaptive_clip_grad(parameters, value, norm_type=norm_type)
	else:
		raise ValueError('invalid clip mode: {}'.format(mode))
	

def get_params(model, names):
	params = []
	for name in names:
		params.extend(list(model.__getattribute__(name).parameters()))
	return params


def get_timepc(cuda_synchronize=False):
	if torch.cuda.is_available() and cuda_synchronize:
		torch.cuda.synchronize()
	return time.perf_counter()


def set_requires_grad(models, requires_grad=False):
	if not isinstance(models, list):
		models = [models]
	for model in models:
		for p in model.parameters():
			p.requires_grad = requires_grad


def print_networks(models, xs, logger):
	models = models if isinstance(models, list) else [models]
	xs = xs if isinstance(xs, list) else [xs]
	for model, x in zip(models, xs):
		result = '\n' + '-' * 36 + ' {} '.format(type(model).__name__) + '-' * 36 + '\n'
		# total_num_params = 0
		# for i, (name, child) in enumerate(model.named_children()):
		# 	num_params = sum([p.numel() for p in child.parameters()]) / 1e6
		# 	total_num_params += num_params
		# 	result += '{}: {:<.3f}M\n'.format(name, num_params)
		# 	for i, (grandname, grandchild) in enumerate(child.named_children()):
		# 		num_params = sum([p.numel() for p in grandchild.parameters()]) / 1e6
		# 		result += '==> {}: {:<3.3f}M\n'.format(grandname, num_params)
		# total_num_params_with_parameter_vars = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
		# result += '[Network {}] Total number of parameters: {:<.3f}M (with parameter_vars: {:<.3f}M)\n'.format(type(model).__name__, total_num_params, total_num_params_with_parameter_vars)
		flops = FlopCountAnalysis(model, x)
		result += '{}\n'.format(flop_count_table(flops, max_depth=5))
		result += '-' * (72 + 2 + len(type(model).__name__))
		log_msg(logger, result)

def reduce_tensor(tensor, world_size, mode='sum', sum_avg=True, rank=0):
	if isinstance(tensor, torch.Tensor):
		tensor_ = tensor.detach()
		if tensor_.device == torch.device('cpu'):
			tensor_ = tensor_.cuda()
	else:
		tensor_ = torch.tensor(tensor).float().cuda()
	if world_size == 1:
		return tensor_
	if mode == 'sum':
		dist.barrier()
		dist.all_reduce(tensor_, op=torch.distributed.ReduceOp.SUM, )
		if sum_avg:
			tensor_ /= world_size
		tensor_out = tensor_
	elif mode == 'cat':
		size = [1] * len(tensor_.shape)
		size[0] = world_size
		tensor_out = torch.zeros_like(tensor_, dtype=tensor_.dtype, device=tensor_.device)
		tensor_out = tensor_out.repeat(size)
		B = tensor_.shape[0]
		tensor_out[rank * B:(rank+1) * B] = tensor_
		dist.barrier()
		dist.all_reduce(tensor_out, op=torch.distributed.ReduceOp.SUM, )
	elif mode == 'and':
		dist.barrier()
		dist.all_reduce(tensor_, op=torch.distributed.ReduceOp.BAND, )
		tensor_out = tensor_
	elif mode == 'or':
		dist.barrier()
		dist.all_reduce(tensor_, op=torch.distributed.ReduceOp.BOR, )
		tensor_out = tensor_
	else:
		raise 'invalid reduce mode: {}'.format(mode)
	return tensor_out


def distribute_bn(model, world_size, dist_bn):
	# ensure every node has the same running bn stats
	model = model.module if hasattr(model, 'module') else model
	for bn_name, bn_buf in model.named_buffers(recurse=True):
		if ('running_mean' in bn_name) or ('running_var' in bn_name):
			if dist_bn == 'reduce':
				# average bn stats across whole group
				torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
				bn_buf /= float(world_size)
			elif dist_bn == 'broadcast':
				# broadcast bn stats from rank 0 to whole group
				torch.distributed.broadcast(bn_buf, 0)
			else:
				pass


def get_loss_scaler(scaler='native'):
	scaler_dict = {
		'none': None,
		'native': NativeScaler(),
		'apex': ApexScaler(),
	}
	return scaler_dict[scaler]


@contextmanager
def placeholder():
	yield


def get_autocast(autocast='native'):
	autocast_dict = {
		'none': placeholder,
		'native': torch.cuda.amp.autocast,
		'apex': placeholder,
	}
	return autocast_dict[autocast]


def get_net_params(net, requires_grad=True):
	num_params = 0
	for param in net.parameters():
		if requires_grad and param.requires_grad:
			num_params += param.numel()
	return num_params
