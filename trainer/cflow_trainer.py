import os
import copy
import glob
import shutil
import datetime
import time
import math

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term, t2np
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup
import torch.nn.functional as F

import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP

try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER

from model.cflow import positionalencoding2d, activation, get_logp, log_theta


@TRAINER.register_module
class CFLOWTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(CFLOWTrainer, self).__init__(cfg)

		self.iter_full = cfg.trainer.meta_epochs * cfg.trainer.sub_epochs * self.cfg.data.train_size
		
		params = []
		for l in range(self.cfg.model.pool_layers):
			params += list(self.net.decoders[l].parameters())
		self.optim = get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)
		
		# optimizer
		# self.optim = torch.optim.Adam(params, lr=self.cfg.trainer.lr)

	def reset(self, isTrain=True):
		self.net.train(mode=isTrain)
		self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
		
	def scheduler_step(self, step):
		self.scheduler.step(step)
		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
		
	def set_input(self, inputs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()
		self.cls_name = inputs['cls_name']
		self.anomaly = inputs['anomaly']
		self.bs = self.imgs.shape[0]
	
	def forward(self):
		self.net(self.imgs)
	

	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			# self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=filter(lambda p: p.requires_grad, self.net.parameters()), create_graph=self.cfg.loss.create_graph)

		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				# dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
				dispatch_clip_grad(filter(lambda p: p.requires_grad, self.net.parameters()), value=self.cfg.loss.clip_grad)
				
			optim.step()
		
	def optimize_parameters(self, train_loss, train_count):
		if self.mixup_fn is not None:
			self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
		with self.amp_autocast():
			self.forward()


			"""TODO这样也能跑,但是不知道为什么慢了很多
			for l, layer in enumerate(self.net.pool_layers):
				FIB, c_r, e_r, dec_idx, perm, E, C, _, _ = self.net.Decoder_forward(l, layer)
				
				for f in range(FIB):
					log_prob, loss_term = self.net.FIB_forward(f, FIB, c_r, e_r, dec_idx, self.net.N, E, C, self.net.model_backbone.dec_arch, perm=perm)
			"""					

			for l, layer in enumerate(self.net.pool_layers):
				# FIB, c_r, e_r, dec_idx, perm, E, C, _, _ = self.net.Decoder_forward(l, layer)
				
				e = activation[layer].detach()  # BxCxHxW
				#
				B, C, H, W = e.size()
				S = H*W
				E = B*S    
				#
				p = positionalencoding2d(self.net.model_backbone.condition_vec, H, W).to(self.imgs.device).unsqueeze(0).repeat(B, 1, 1, 1)
				c_r = p.reshape(B, self.net.model_backbone.condition_vec, S).transpose(1, 2).reshape(E, self.net.model_backbone.condition_vec)  # BHWxP
				e_r = e.reshape(B, C, S).transpose(1, 2).reshape(E, C)  # BHWxC
				perm = torch.randperm(E).to(self.imgs.device)  # BHW
				# decoder = self.decoders[dec_idx]
				
				FIB = E//self.net.N  # number of fiber batches
				# print('******************************Training FIBint:{}*************************'.format(int(E%self.net.N > 0)))

				for f in range(FIB):
					# log_prob, loss_term = self.net.FIB_forward(f, FIB, c_r, e_r, dec_idx, self.net.N, E, C, self.net.model_backbone.dec_arch, perm=perm)

					idx = torch.arange(f*self.net.N, (f+1)*self.net.N)
					# else:
					# 	idx = torch.arange(f*self.net.model_backbone.N, E)
					
					c_p = c_r[perm[idx]]  # NxP
					e_p = e_r[perm[idx]]  # NxC

					if 'cflow' in self.net.model_backbone.dec_arch:
						z, log_jac_det = self.net.decoders[l](e_p, [c_p,])
					else:
						z, log_jac_det = self.net.decoders[l](e_p)

					decoder_log_prob = get_logp(C, z, log_jac_det)
					log_prob = decoder_log_prob / C  # likelihood per dim
					loss_term = -log_theta(log_prob)
					

					# import pdb;pdb.set_trace()
					self.backward_term(loss_term.mean(), self.optim)

					train_loss += t2np(loss_term.sum())
					train_count += len(loss_term)

					# print('*********************************Train count :{}*********************************'.format(train_count))

		# TODO, To check
		update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_term.mean(), self.world_size).clone().detach().item(), 1, self.master)

		return train_loss, train_count

	def adjust_learning_rate(self, c, optimizer, epoch):
		lr = c.lr
		if c.lr_cosine:
			eta_min = lr * (c.lr_decay_rate ** 3)
			lr = eta_min + (lr - eta_min) * (
						1 + math.cos(math.pi * epoch / c.meta_epochs)) / 2
		else:
			steps = np.sum(epoch >= np.asarray(c.lr_decay_epochs))
			if steps > 0:
				lr = lr * (c.lr_decay_rate ** steps)

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def warmup_learning_rate(self, c, epoch, batch_id, total_batches, optimizer):
		if c.lr_warm and epoch < c.lr_warm_epochs:
			p = (batch_id + epoch * total_batches) / \
				(c.lr_warm_epochs * total_batches)
			lr = c.lr_warmup_from + p * (c.lr_warmup_to - c.lr_warmup_from)
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr
		#
		for param_group in optimizer.param_groups:
			lrate = param_group['lr']
		return lrate

	def _finish(self):
		log_msg(self.logger, 'finish training')
		self.writer.close() if self.master else None
		metric_list = []
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				metric_list.append(self.metric_recorder[f'{metric}_{cls_name}'])
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					metric_list.append(self.metric_recorder[f'{metric}_Avg'])
		f = open(f'{self.cfg.logdir}/metric.txt', 'w')
		msg = ''
		for i in range(len(metric_list[0])):
			for j in range(len(metric_list)):
				msg += '{:3.5f}\t'.format(metric_list[j][i])
			msg += '\n'
		f.write(msg)
		f.close()


	def train(self):
		self.reset(isTrain=True)
		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None

		# for name, params in self.net.decoders[0].named_parameters():
		# 	print(params.requires_grad)
		
		# import pdb;pdb.set_trace() # chceck params' grad
		for meta_epoch in range(self.epoch_full):

			train_length = self.cfg.data.train_size
			train_loader = iter(self.train_loader)
		
			self.adjust_learning_rate(self.cfg.trainer, self.optim, meta_epoch)

			for sub_epoch in range(self.cfg.trainer.sub_epochs):
				train_loss = 0.0
				train_count = 0

				for i in range(train_length):

					lr = self.warmup_learning_rate(self.cfg.trainer, meta_epoch, i+sub_epoch*train_length, train_length*self.cfg.trainer.sub_epochs, self.optim)
					# print('*********************************[sub_epoch / epoch] : [{} / {}] , learning rate :{} ********************************'.format(sub_epoch, meta_epoch, lr))
					# self.scheduler_step(self.iter) 
					# ---------- data ----------
					t1 = get_timepc()
					self.iter += 1
					train_data = next(train_loader)
					self.set_input(train_data)
					t2 = get_timepc()
					update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
					# ---------- optimization ----------
					# import pdb;pdb.set_trace()
					train_loss, train_count = self.optimize_parameters(train_loss, train_count)

					t3 = get_timepc()
					update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
					update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
					# ---------- log ----------
					if self.master:
						if self.iter % self.cfg.logging.train_log_per == 0:
							msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
							log_msg(self.logger, msg)
							if self.writer:
								for k, v in self.log_terms.items():
									self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
								self.writer.flush()
					if self.iter % self.cfg.logging.train_reset_log_per == 0:
						self.reset(isTrain=True)
					# ---------- update train_loader ----------
					if self.iter % train_length == 0:
						self.epoch += 1
						# import pdb;pdb.set_trace()
						if self.cfg.dist and self.dist_BN != '':
							distribute_bn(self.net, self.world_size, self.dist_BN)
						self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
						if self.epoch >= self.cfg.trainer.test_start_epoch or self.epoch % self.cfg.trainer.test_per_epoch == 0:
							self.test()
						else:
							self.test_ghost()
						self.cfg.total_time = get_timepc() - self.cfg.task_start_time
						total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
						eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
						log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
						self.save_checkpoint()
						self.reset(isTrain=True)
						self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
						train_loader = iter(self.train_loader)
					
				mean_train_loss = train_loss / train_count
				print('Epoch: {:d}.{:d} \t train loss: {:.4f}, lr={:.6f}'.format(meta_epoch, sub_epoch, mean_train_loss, lr))
		
		
		self._finish()


	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)

	@torch.no_grad()
	def test(self):
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
		test_dist = [list() for _ in self.net.pool_layers]
		batch_idx = 0
		test_loss = 0.0
		test_count = 0
		test_length = self.cfg.data.test_size
		test_loader = iter(self.test_loader)

		while batch_idx < test_length:
			# if batch_idx == 10:
			# 	break
			height = list()
			width = list()
		
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)
			self.set_input(test_data)
			self.forward()

			for l, layer in enumerate(self.net.pool_layers):
				FIB, c_r, e_r, dec_idx, _, E, C, H, W = self.net.Decoder_forward(l, layer)
				height.append(H)
				width.append(W)
				# print('******************************Testing FIBint:{}*************************'.format(int(E%self.net.N > 0)))

				# print('**********************************height :{} , width:{}**********************************'.format(height, width))
				for f in range(FIB):
					log_prob, loss_term = self.net.FIB_forward(f, FIB, c_r, e_r, dec_idx, self.net.N, E, C, self.net.model_backbone.dec_arch)
					test_loss += t2np(loss_term.sum())
					test_count += len(loss_term)
					test_dist[l] = test_dist[l] + log_prob.detach().cpu().tolist()
			update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_term.mean(), self.world_size).clone().detach().item(), 1, self.master)
			
			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			cls_names.append(np.array(self.cls_name))
			anomalys.append(self.anomaly.cpu().numpy().astype(int))
			
			t2 = get_timepc()

			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
			# ---------- log ----------
			if self.master:
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
					log_msg(self.logger, msg)

		mean_test_loss = test_loss / test_count
		print('Epoch: {:d} \t test_loss: {:.4f}'.format(self.epoch, mean_test_loss))
		#

		# get anomaly maps
		test_map = [list() for p in self.net.pool_layers]
		for l, p in enumerate(self.net.pool_layers):
			test_norm = torch.tensor(test_dist[l], dtype=torch.double)  # EHWx1
			test_norm-= torch.max(test_norm) # normalize likelihoods to (-Inf:0] by subtracting a constant
			test_prob = torch.exp(test_norm) # convert to probs in range [0:1]
			test_mask = test_prob.reshape(-1, height[l], width[l])
			test_mask = test_prob.reshape(-1, height[l], width[l])
			# upsample
			test_map[l] = F.interpolate(test_mask.unsqueeze(1),
				size=self.cfg.size, mode='bilinear', align_corners=True).squeeze().numpy()
		# score aggregation
		score_map = np.zeros_like(test_map[0])
		for l, p in enumerate(self.net.pool_layers):
			score_map += test_map[l]
		score_mask = score_map
		# invert probs to anomaly scores
		anomaly_map = score_mask.max() - score_mask
		anomaly_maps.append(anomaly_map)

		# merge results
		if self.cfg.dist:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
			if self.master:
				results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
				valid_results = False
				while not valid_results:
					results_files = glob.glob(f'{self.tmp_dir}/*.pth')
					if len(results_files) != self.cfg.world_size:
						time.sleep(1)
					else:
						idx_result = 0
						while idx_result < self.cfg.world_size:
							results_file = results_files[idx_result]
							try:
								result = torch.load(results_file)
								for k, v in result.items():
									results[k].extend(v)
								idx_result += 1
							except:
								time.sleep(1)
						valid_results = True
		else:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
		if self.master:
			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
			msg = {}
			for idx, cls_name in enumerate(self.cls_names):
				metric_results = self.evaluator.run(results, cls_name, self.logger)
				msg['Name'] = msg.get('Name', [])
				msg['Name'].append(cls_name)
				avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
				msg['Name'].append('Avg') if avg_act else None
				# msg += f'\n{cls_name:<10}'
				for metric in self.metrics:
					metric_result = metric_results[metric] * 100
					self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
					max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
					max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
					msg[metric] = msg.get(metric, [])
					msg[metric].append(metric_result)
					msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
					if avg_act:
						metric_result_avg = sum(msg[metric]) / len(msg[metric])
						self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
						max_metric = max(self.metric_recorder[f'{metric}_Avg'])
						max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
						msg[metric].append(metric_result_avg)
						msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
			msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center",
									stralign="center", )
			log_msg(self.logger, f'\n{msg}')
