import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as F_tv
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from . import LOSS
from model import get_model

__all__ = ['CE', 'LabelSmoothingCE', 'SoftTargetCE', 'CLSKDLoss']


@LOSS.register_module
class CE(nn.CrossEntropyLoss):
	def __init__(self, lam=1):
		super(CE, self).__init__()
		self.lam = lam

	def forward(self, input, target):
		return super(CE, self).forward(input, target) * self.lam


@LOSS.register_module
class LabelSmoothingCE(nn.Module):
	"""
	NLL loss with label smoothing.
	"""
	def __init__(self, smoothing=0.1, lam=1):
		"""
		Constructor for the LabelSmoothing module.
		:param smoothing: label smoothing factor
		"""
		super(LabelSmoothingCE, self).__init__()
		assert smoothing < 1.0
		self.smoothing = smoothing
		self.lam = lam
		self.confidence = 1. - smoothing

	def forward(self, x, target):
		logprobs = F.log_softmax(x, dim=-1)
		nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
		nll_loss = nll_loss.squeeze(1)
		smooth_loss = -logprobs.mean(dim=-1)
		loss = self.confidence * nll_loss + self.smoothing * smooth_loss
		return loss.mean() * self.lam


@LOSS.register_module
class SoftTargetCE(nn.Module):
	def __init__(self, lam=1, fp32=False):
		super(SoftTargetCE, self).__init__()
		self.lam = lam
		self.fp32 = fp32

	def forward(self, x, target):
		if self.fp32:
			loss = torch.sum(-target * F.log_softmax(x.float(), dim=-1), dim=-1)
		else:
			loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
		return loss.mean() * self.lam


@LOSS.register_module
class CLSKDLoss(torch.nn.Module):
	def __init__(self, cfg, kd_type='soft', size=224, mean_t=IMAGENET_DEFAULT_MEAN, std_t=IMAGENET_DEFAULT_STD,
				 mean_s=IMAGENET_DEFAULT_MEAN, std_s=IMAGENET_DEFAULT_STD, tau=1.0, lam=1):
		super().__init__()
		self.teacher_model = get_model(cfg)
		self.teacher_model.cuda()
		self.teacher_model.eval()
		assert kd_type in ['soft', 'hard']
		self.kd_type = kd_type
		self.size = size
		self.mean_t, self.std_t = mean_t, std_t
		self.mean_s, self.std_s = mean_s, std_s
		self.tau = tau
		self.lam = lam

	def forward(self, outputs_kd, inputs):
		with torch.no_grad():
			if self.mean_t != self.mean_s or self.std_t != self.std_s:
				# std = [std_t / std_s for std_t, std_s in zip(self.std_t, self.std_s)]
				# transform_std = T.Normalize(self.mean_t, std=std)
				# mean = [mean_t / mean_s for mean_t, mean_s in zip(self.mean_t, self.mean_s)]
				# transform_mean = T.Normalize(mean=mean, std=self.std_t)
				# inputs = transform_mean(transform_std(inputs))
				mean_t = torch.as_tensor(self.mean_t, dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1)
				std_t = torch.as_tensor(self.std_t, dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1)
				mean_s = torch.as_tensor(self.mean_s, dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1)
				std_s = torch.as_tensor(self.std_s, dtype=inputs.dtype, device=inputs.device).view(-1, 1, 1)
				inputs = inputs.clone()
				inputs.mul_(std_s).add_(mean_s).sub_(mean_t).div_(std_t)
			B, C, H, W = inputs.shape
			if H != self.size:
				inputs = F_tv.resize(inputs, self.size, F_tv.InterpolationMode.BICUBIC)
			teacher_outputs = self.teacher_model(inputs)
		if self.kd_type == 'soft':
			distillation_loss = F.kl_div(F.log_softmax(outputs_kd / self.tau, dim=1),
										 F.log_softmax(teacher_outputs / self.tau, dim=1),
										 reduction='sum', log_target=True) * (self.tau * self.tau) / outputs_kd.shape[0]
		elif self.kd_type == 'hard':
			distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
		else:
			raise ValueError(f'invalid distillation type: {self.kd_type}')
		return distillation_loss * self.lam
