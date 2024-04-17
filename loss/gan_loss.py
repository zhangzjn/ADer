import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from . import LOSS

__all__ = ['GANLoss', 'GPLoss', 'R1Loss', 'PathLoss']


@LOSS.register_module
class GANLoss(nn.Module):
	def __init__(self, mode='hinge',
				 change_label_p=0.0,
				 one_side_label_smooth=0.0,
				 lam=1):
		super(GANLoss, self).__init__()
		self.mode = mode
		self.change_label_p = change_label_p
		self.one_side_label_smooth = one_side_label_smooth
		self.lam = lam
		if self.mode not in ['bce', 'mse', 'hinge', 'wgan', 'logistic_saturating', 'logistic_nonsaturating', 'relativistic_gan']:
			raise NotImplementedError('gan loss {} is not implemented'.format(self.mode))

	def get_target_tensor(self, pred, tgt):
		shape = pred.shape
		# tgt = torch.full((B,), tgt, dtype=pred.dtype)
		tgt = torch.full((shape), tgt, dtype=pred.dtype)
		# random change label
		if self.change_label_p >= 0.0:
			is_not_change = (torch.rand(shape) > self.change_label_p)
			is_not_change = is_not_change.float()
			tgt = tgt * is_not_change + (1 - tgt) * (1 - is_not_change)  # xnor
		# one side label smooth
		if self.one_side_label_smooth >= 0.0:
			# tgt_tensor = (tgt * 1 - torch.rand(shape) * self.one_side_label_smooth).abs()  # [0~0.1, 0.9~1]
			tgt_tensor = (tgt * 1 - torch.rand(shape) * self.one_side_label_smooth) * tgt  # [0, 0.9~1]
			# tgt_tensor = torch.max(tgt * 1 - torch.rand(B) * self.label_smooth)  # to be modify: only applying to real image.
		else:
			tgt_tensor = tgt * 1
		# return tgt_tensor.cuda(pred.device)
		return tgt_tensor.expand_as(pred).cuda(pred.device)

	def call_one(self, pred, should_be_classified_as_real):
		if self.mode == 'logistic_nonsaturating':
			loss = F.softplus(-pred).mean() if should_be_classified_as_real else F.softplus(pred).mean()
		else:
			raise 'invalid loss mode: {}'.format(self.loss_mode)
		return loss

	def __call__(self, pred_fake=None, pred_real=None, isD=True):
		if pred_fake is None:
			raise ValueError('meaningless input for GAN loss')
		loss = 0
		if self.mode == 'bce':
			if isD:
				loss_real = nn.BCEWithLogitsLoss()(pred_real, self.get_target_tensor(pred_real, 1.0))
				loss_fake = nn.BCEWithLogitsLoss()(pred_fake, self.get_target_tensor(pred_fake, 0.0))
				loss = loss_real + loss_fake
			else:
				loss_fake = nn.BCEWithLogitsLoss()(pred_fake, self.get_target_tensor(pred_fake, 1.0))
				loss = loss_fake
		elif self.mode == 'mse':
			if isD:
				loss_real = nn.MSELoss()(pred_real, self.get_target_tensor(pred_real, 1.0))
				loss_fake = nn.MSELoss()(pred_fake, self.get_target_tensor(pred_fake, 0.0))
				loss = loss_real + loss_fake
			else:
				loss_fake = nn.MSELoss()(pred_fake, self.get_target_tensor(pred_fake, 1.0))
				loss = loss_fake
		elif self.mode == 'hinge':
			if isD:
				loss_real = nn.ReLU()(1.0 - pred_real).mean()
				loss_fake = nn.ReLU()(1.0 + pred_fake).mean()
				loss = loss_real + loss_fake
			else:
				loss_fake = -pred_fake.mean()
				loss = loss_fake
		elif self.mode == 'wgan':
			if isD:
				loss_real = -pred_real.mean()
				loss_fake = pred_fake.mean()
				loss = loss_real + loss_fake
			else:
				loss_fake = -pred_fake.mean()
				loss = loss_fake
		elif self.mode == 'logistic_saturating':
			if isD:
				loss_real = F.softplus(-pred_real).mean()  # log(1+exp(x))
				loss_fake = F.softplus(pred_fake).mean()
				loss = loss_real + loss_fake
			else:
				loss_fake = -F.softplus(pred_fake).mean()
				loss = loss_fake
		elif self.mode == 'logistic_nonsaturating':
			if isD:
				loss_real = F.softplus(-pred_real).mean()  # log(1+exp(x))
				loss_fake = F.softplus(pred_fake).mean()
				loss = loss_real + loss_fake
			else:
				loss_fake = F.softplus(-pred_fake).mean()
				loss = loss_fake
		elif self.mode == 'relativistic_gan':
			if isD:
				loss_real = nn.BCEWithLogitsLoss()(pred_real - pred_fake.mean(0, keepdim=True), torch.ones_like(pred_real))
				loss_fake = nn.BCEWithLogitsLoss()(pred_fake - pred_real.mean(0, keepdim=True), torch.zeros_like(pred_real))
				loss = loss_real + loss_fake
			else:
				loss_fake = nn.BCEWithLogitsLoss()(pred_fake - pred_real.mean(0, keepdim=True), torch.ones_like(pred_real))
				loss = loss_fake
		return loss * self.lam


@LOSS.register_module
class GPLoss(nn.Module):
	def __init__(self,
				 lam=1):
		super(GPLoss, self).__init__()
		self.lam = lam

	def forward(self, netD, real_data, fake_data):
		batch_size = real_data.size()[0]
		LAMBDA = 1
		alpha = torch.rand(batch_size, 1, 1, 1)
		alpha = alpha.expand_as(real_data).cuda()
		interpolates = alpha * real_data + (1 - alpha) * fake_data
		interpolates = interpolates.cuda()
		interpolates = autograd.Variable(interpolates, requires_grad=True)
		disc_interpolates = netD(interpolates)
		gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
								  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
								  create_graph=True, retain_graph=True, only_inputs=True)[0]
		gradients = gradients.view(gradients.size(0), -1)
		gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
		return gradient_penalty * self.lam


@LOSS.register_module
class R1Loss(nn.Module):
	def __init__(self, lam=1):
		super(R1Loss, self).__init__()
		self.lam = lam
	
	def forward(self, images, output):
		gradients = autograd.grad(outputs=output, inputs=images, grad_outputs=torch.ones(output.size(), device=images.device),
								  create_graph=True, retain_graph=True, only_inputs=True)[0].view(images.size(0), -1)
		r1_penalty = torch.sum(gradients.pow(2)).mean()
		
		# with no_weight_gradients():
		# 	grad_real, = autograd.grad(
		# 		outputs=output.sum(), inputs=images, create_graph=True
		# 	)
		# r1_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
		return r1_penalty * self.lam
	
	
@LOSS.register_module
class PathLoss(nn.Module):
	def __init__(self,
				 lam=1):
		super(PathLoss, self).__init__()
		self.lam = lam

	def forward(self, img_fake, latent, mean_path_length, decay=0.01):
		noise = torch.randn_like(img_fake) / math.sqrt(img_fake.shape[2] * img_fake.shape[3])
		grad, = autograd.grad(outputs=(img_fake * noise).sum(), inputs=latent, create_graph=True)
		path_length = torch.sqrt(grad.pow(2).sum(2).mean(1))
		mean_path_length_out = mean_path_length + decay * (path_length.mean() - mean_path_length)
		path_penalty = (path_length - mean_path_length_out).pow(2).mean()
		return path_penalty * self.lam, path_length, mean_path_length_out.detach()


